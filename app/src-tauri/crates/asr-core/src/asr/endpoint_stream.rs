//! 발화 단위 엔드포인트 스트리밍 — 윈도우 재디코드 대신 "발화 하나당 전사 1회".
//!
//! VAD 로 발화 경계(무음 hangover)를 잡아, 발화가 끝날 때마다 그 발화만 whisper 로 한 번
//! 전사한다. 성장 버퍼 재디코드 / 공통접두 확정 / 중복제거 기계장치가 전부 사라져 CPU 가 낮고
//! "지금"을 따라가며, 확정이 발화 끝마다 문장 단위로 한 번에 나온다(anarlog audio-chunking 의
//! 발화 엔드포인팅을 에너지 VAD 로 이식한 버전).
//!
//! VAD 는 `Box<dyn Vad>` 뒤에 있어 EnergyVad(현재)든 신경망 VAD(추후 sherpa SileroVad)든 교체가
//! 생성자 한 줄이다. `transcribe_full` 은 SelfStreamingBackend 를 재사용하므로 Whisper/SenseVoice/
//! Qwen 어디에나 붙는다.

use std::collections::VecDeque;

use async_trait::async_trait;

use super::backend::{AsrConfig, BackendCaps, StreamingAsrBackend};
use super::self_stream::SelfStreamingBackend;
use super::token::{AsrError, AsrToken};
use super::wordtime;
use crate::vad::Vad;

const SR: f64 = 16_000.0;
/// VAD 프레임(샘플). 512=32ms — 추후 Silero(512 고정 프레임)와 그대로 맞춘다.
const FRAME: usize = 512;

/// 엔드포인트 판정 노브(전부 env 로 재빌드 없이 튜닝 가능).
#[derive(Clone, Copy, Debug)]
pub struct EndpointConfig {
    /// 발화 뒤 이만큼(초) 연속 무음이면 발화 종료로 보고 확정한다.
    pub hangover_sec: f64,
    /// 이보다 짧은 (무음 트림 후) 발화는 무시(기침/클릭/오검출).
    pub min_utt_sec: f64,
    /// 이보다 길면 강제 종료(긴 독백도 확정이 나오게).
    pub max_utt_sec: f64,
    /// 발화 시작 앞에 붙일 프리롤(초) — 첫 음절이 잘리지 않게.
    pub pre_roll_sec: f64,
    /// 종료 시 남길 뒤쪽 무음(초). 나머지 무음은 잘라 타임스탬프를 조인다.
    pub keep_trail_sec: f64,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            hangover_sec: 0.6,
            min_utt_sec: 0.3,
            max_utt_sec: 20.0,
            pre_roll_sec: 0.3,
            keep_trail_sec: 0.2,
        }
    }
}

impl EndpointConfig {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Some(v) = env_f64("ENDPOINT_HANGOVER_SEC") {
            c.hangover_sec = v;
        }
        if let Some(v) = env_f64("ENDPOINT_MIN_UTT_SEC") {
            c.min_utt_sec = v;
        }
        if let Some(v) = env_f64("ENDPOINT_MAX_UTT_SEC") {
            c.max_utt_sec = v;
        }
        if let Some(v) = env_f64("ENDPOINT_PRE_ROLL_SEC") {
            c.pre_roll_sec = v;
        }
        if let Some(v) = env_f64("ENDPOINT_KEEP_TRAIL_SEC") {
            c.keep_trail_sec = v;
        }
        c
    }
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
}

/// 발화 엔드포인트 드라이버. StreamingAsrBackend 로 노출되어 기존 run_session 드라이버에
/// 그대로 꽂힌다(insert_audio_chunk 로 오디오 공급, process_iter 로 확정 폴링).
pub struct EndpointStreamingProcessor<B: SelfStreamingBackend> {
    backend: B,
    vad: Box<dyn Vad>,
    cfg: EndpointConfig,
    /// 아직 FRAME 단위로 처리 못 한 입력 tail.
    inbox: Vec<f32>,
    /// inbox[0] 의 절대 시각(초). None=아직 입력 없음.
    next_time: Option<f64>,
    in_speech: bool,
    /// 현재 발화 누적(프리롤 포함).
    utt: Vec<f32>,
    /// 현재 발화 절대 시작 시각(초).
    utt_start: f64,
    /// 현재 발화 끝 연속 무음(샘플).
    trailing_silence: usize,
    /// 무음 중 굴리는 프리롤 링버퍼.
    pre_roll: VecDeque<f32>,
}

impl<B: SelfStreamingBackend> EndpointStreamingProcessor<B> {
    pub fn new(backend: B, vad: Box<dyn Vad>) -> Self {
        Self::with_config(backend, vad, EndpointConfig::from_env())
    }

    pub fn with_config(backend: B, vad: Box<dyn Vad>, cfg: EndpointConfig) -> Self {
        Self {
            backend,
            vad,
            cfg,
            inbox: Vec::new(),
            next_time: None,
            in_speech: false,
            utt: Vec::new(),
            utt_start: 0.0,
            trailing_silence: 0,
            pre_roll: VecDeque::new(),
        }
    }

    fn reset_utt(&mut self) {
        self.in_speech = false;
        self.utt.clear();
        self.trailing_silence = 0;
        self.pre_roll.clear();
    }

    /// 완성된 발화를 1회 전사 → 토큰. (무음 트림 후) 너무 짧으면 빈 벡터.
    fn finalize(&mut self) -> Result<Vec<AsrToken>, AsrError> {
        // 종료 시 뒤쪽 무음 대부분을 잘라 타임스탬프를 조인다(keep_trail 만 남김).
        let keep_trail = (self.cfg.keep_trail_sec * SR) as usize;
        let trim = self
            .trailing_silence
            .saturating_sub(keep_trail)
            .min(self.utt.len());
        let end = self.utt.len() - trim;
        let dur = end as f64 / SR;
        let start = self.utt_start;
        if dur < self.cfg.min_utt_sec {
            return Ok(Vec::new());
        }
        // utt 를 소유로 꺼내 backend(&mut) 와의 borrow 충돌을 피한다.
        let utt = std::mem::take(&mut self.utt);
        // prompt "" — 발화 간 프롬프트 이월 금지(한국어 환각/중복 유발, self_stream 과 동일 규칙).
        let text = self.backend.transcribe_full(&utt[..end], "")?;
        Ok(split_tokens(&text, start, dur))
    }

    /// inbox 를 FRAME 단위로 소진하며 상태머신을 돌린다. 종료된 발화 토큰을 모아 반환.
    fn drain_frames(&mut self, flush: bool) -> Result<Vec<AsrToken>, AsrError> {
        let mut out = Vec::new();
        let max_pre = (self.cfg.pre_roll_sec * SR) as usize;
        let hangover = (self.cfg.hangover_sec * SR) as usize;
        let max_utt = (self.cfg.max_utt_sec * SR) as usize;

        while self.inbox.len() >= FRAME {
            let frame: Vec<f32> = self.inbox.drain(..FRAME).collect();
            let t0 = self.next_time.unwrap_or(0.0);
            let frame_end = t0 + FRAME as f64 / SR;
            self.next_time = Some(frame_end);
            let speech = self.vad.is_speech(&frame);

            if !self.in_speech {
                if speech {
                    // 발화 시작: 프리롤(직전 무음 pad) + 현재 프레임.
                    self.utt = self.pre_roll.drain(..).collect();
                    self.utt.extend_from_slice(&frame);
                    self.utt_start = frame_end - self.utt.len() as f64 / SR;
                    self.trailing_silence = 0;
                    self.in_speech = true;
                } else {
                    self.pre_roll.extend(frame.iter().copied());
                    while self.pre_roll.len() > max_pre {
                        self.pre_roll.pop_front();
                    }
                }
            } else {
                self.utt.extend_from_slice(&frame);
                if speech {
                    self.trailing_silence = 0;
                } else {
                    self.trailing_silence += FRAME;
                }
                if self.trailing_silence >= hangover || self.utt.len() >= max_utt {
                    out.extend(self.finalize()?);
                    self.reset_utt();
                }
            }
        }

        // 스트림 종료(flush): 남은 부분 프레임을 발화에 붙이고 진행 중 발화를 확정.
        if flush {
            if self.in_speech {
                self.utt.extend_from_slice(&self.inbox);
                out.extend(self.finalize()?);
                self.reset_utt();
            }
            self.inbox.clear();
        }
        Ok(out)
    }
}

/// 발화 텍스트를 [start, start+dur] 구간에 글자수 비례로 토큰화(self_stream 과 동일 아이디어).
fn split_tokens(text: &str, start: f64, dur: f64) -> Vec<AsrToken> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let weights: Vec<f64> = words.iter().map(|w| wordtime::time_weight(w)).collect();
    let total_w: f64 = weights.iter().sum::<f64>().max(1.0);
    let mut cum = vec![0.0f64; words.len() + 1];
    for j in 0..words.len() {
        cum[j + 1] = cum[j] + weights[j];
    }
    words
        .iter()
        .enumerate()
        .map(|(j, w)| {
            let s = start + dur * (cum[j] / total_w);
            let e = start + dur * (cum[j + 1] / total_w);
            let text = if j > 0 {
                format!(" {}", w)
            } else {
                (*w).to_string()
            };
            AsrToken {
                start: s,
                end: e,
                text,
                probability: None,
                detected_language: None,
                speaker: None,
            }
        })
        .collect()
}

#[async_trait]
impl<B: SelfStreamingBackend> StreamingAsrBackend for EndpointStreamingProcessor<B> {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        self.backend.configure(cfg)
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        self.backend.warmup()
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64) {
        if self.next_time.is_none() {
            self.next_time = Some((end_time - pcm.len() as f64 / SR).max(0.0));
        }
        self.inbox.extend_from_slice(pcm);
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        self.drain_frames(is_last)
    }

    fn get_buffer(&self) -> String {
        // 발화 확정 전까지는 텍스트가 없다(발화당 1회 전사). 진행 중 partial 없음.
        String::new()
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.backend.set_language(lang);
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps {
            provides_word_timestamps: false,
            provides_probability: false,
            self_streaming: true,
            tokenizer_id: "endpoint_stream",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::Vad;

    /// 진폭 0.1 초과 프레임을 발화로 보는 테스트용 VAD.
    struct AmpVad;
    impl Vad for AmpVad {
        fn is_speech(&mut self, samples: &[f32]) -> bool {
            samples.iter().any(|s| s.abs() > 0.1)
        }
    }

    struct FixedBackend;
    impl SelfStreamingBackend for FixedBackend {
        fn configure(&mut self, _cfg: &AsrConfig) -> Result<(), AsrError> {
            Ok(())
        }
        fn transcribe_full(&mut self, _samples: &[f32], _prompt: &str) -> Result<String, AsrError> {
            Ok("안녕 하세요".into())
        }
        fn set_language(&mut self, _lang: Option<String>) {}
    }

    fn cfg_fast() -> EndpointConfig {
        EndpointConfig {
            hangover_sec: 0.064, // 2 프레임
            min_utt_sec: 0.03,
            max_utt_sec: 20.0,
            pre_roll_sec: 0.0,
            keep_trail_sec: 0.0,
        }
    }

    #[tokio::test]
    async fn endpoint_emits_after_hangover_silence() {
        let mut p =
            EndpointStreamingProcessor::with_config(FixedBackend, Box::new(AmpVad), cfg_fast());
        p.configure(&AsrConfig::default()).await.unwrap();
        // 5 프레임 발화(0.2) + 3 프레임 무음(0.0) → hangover(2프레임) 초과 → 확정.
        let mut pcm = vec![0.2f32; FRAME * 5];
        pcm.extend(vec![0.0f32; FRAME * 3]);
        let end_time = pcm.len() as f64 / SR;
        p.insert_audio_chunk(&pcm, end_time);

        let toks = p.process_iter(false).await.unwrap();
        let text: String = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(text, "안녕 하세요");
        assert!(toks[0].start >= 0.0);
        assert!(toks.last().unwrap().end <= end_time + 0.001);
    }

    #[tokio::test]
    async fn no_commit_while_still_speaking_then_flush() {
        let mut p =
            EndpointStreamingProcessor::with_config(FixedBackend, Box::new(AmpVad), cfg_fast());
        p.configure(&AsrConfig::default()).await.unwrap();
        // 발화만, 뒤 무음 없음 → 발화 진행 중이라 확정 안 나옴.
        let pcm = vec![0.2f32; FRAME * 5];
        p.insert_audio_chunk(&pcm, pcm.len() as f64 / SR);
        let toks = p.process_iter(false).await.unwrap();
        assert!(toks.is_empty(), "발화 중엔 확정이 나오면 안 된다");

        // 정지(flush) 시 진행 중 발화를 확정.
        let toks2 = p.process_iter(true).await.unwrap();
        let text: String = toks2.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(text, "안녕 하세요");
    }
}
