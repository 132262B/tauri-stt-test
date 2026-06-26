//! 자체 스트리밍 ASR 래퍼 (발화 단위 모델용) — docs/02-architecture.md D.5.
//!
//! SenseVoice/Whisper/Qwen 등 "버퍼 전체 → 텍스트" 모델을 단어 prefix-commit(연속 2회
//! 전사의 최장 공통 접두사 확정)으로 StreamingAsrBackend 인터페이스에 맞춘다.
//!
//! 실시간 최적화:
//! - 윈도우를 **무음 경계에서** 리셋해 윈도우 끝의 남은 단어를 빨리 확정(지연↓)하고
//!   단어 중간이 잘리는 걸 줄인다. 무음이 없으면 MAX_WINDOW 에서 강제 리셋.
//! 타임스탬프는 윈도우 비례 근사.
//! (whisper initial_prompt 로 직전 텍스트를 넘기는 방식은 환각/중복을 유발해 쓰지 않는다.)

use async_trait::async_trait;

use super::backend::{AsrConfig, AsrProfile, BackendCaps, StreamingAsrBackend};
use super::token::{AsrError, AsrToken};

const SR: f64 = 16_000.0;
/// 리셋 가능한 "무음" 판정 RMS 임계.
const SILENCE_RMS: f32 = 0.012;
/// 무음 판정에 보는 버퍼 끝부분 길이(초).
const SILENCE_TAIL_SEC: f64 = 0.4;
/// 실제 마이크 발화는 회의 파일보다 RMS가 작을 수 있으므로, 디코더 실행 게이트는
/// 윈도우 리셋용 silence 기준보다 낮게 둔다.
const ACTIVE_RMS: f32 = 0.0015;
/// 이보다 짧은 활성 음성은 디코더를 돌리지 않는다.
const MIN_ACTIVE_SEC: f64 = 0.25;
/// 활성 음성 판정 프레임 길이(초).
const ENERGY_FRAME_SEC: f64 = 0.02;

#[derive(Clone, Copy, Debug)]
struct SelfStreamingOptions {
    target_window_samples: usize,
    max_window_samples: usize,
    silence_rms: f32,
    active_rms: f32,
    silence_tail_samples: usize,
    min_active_samples: usize,
    energy_frame_samples: usize,
}

impl SelfStreamingOptions {
    fn balanced() -> Self {
        Self::from_secs(12, 20)
    }

    fn realtime_q5() -> Self {
        Self::from_secs(8, 12)
    }

    fn from_config(cfg: &AsrConfig) -> Self {
        let base = match cfg.effective_profile() {
            AsrProfile::RealtimeQ5 => Self::realtime_q5(),
            AsrProfile::Auto | AsrProfile::Balanced => Self::balanced(),
        };
        base.with_env_overrides()
    }

    fn from_secs(target: usize, max: usize) -> Self {
        Self {
            target_window_samples: target * 16_000,
            max_window_samples: max * 16_000,
            silence_rms: SILENCE_RMS,
            active_rms: ACTIVE_RMS,
            silence_tail_samples: (SILENCE_TAIL_SEC * SR) as usize,
            min_active_samples: (MIN_ACTIVE_SEC * SR) as usize,
            energy_frame_samples: (ENERGY_FRAME_SEC * SR) as usize,
        }
    }

    fn with_env_overrides(mut self) -> Self {
        if let Some(sec) = std::env::var("SELF_WIN_SEC")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            self.target_window_samples = sec * 16_000;
            self.max_window_samples = self.target_window_samples * 5 / 3;
        }
        if let Some(sec) = std::env::var("SELF_MAX_WIN_SEC")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            self.max_window_samples = sec * 16_000;
        }
        if let Some(rms) = std::env::var("SELF_SILENCE_RMS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            self.silence_rms = rms;
        }
        if let Some(rms) = std::env::var("SELF_ACTIVE_RMS")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
        {
            self.active_rms = rms;
        }
        self
    }
}

/// "16kHz mono 버퍼 → 전체 텍스트" 동기 백엔드(SenseVoice/Whisper/Qwen 등).
/// prompt: 직전 확정 텍스트(연속성용). 지원 안 하면 무시 가능.
pub trait SelfStreamingBackend: Send {
    fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError>;
    fn warmup(&mut self) -> Result<(), AsrError> {
        Ok(())
    }
    fn transcribe_full(&mut self, samples: &[f32], prompt: &str) -> Result<String, AsrError>;
    fn set_language(&mut self, lang: Option<String>);
}

pub struct SelfStreamingProcessor<B: SelfStreamingBackend> {
    backend: B,
    audio: Vec<f32>,
    offset: f64,
    committed_n: usize,
    last_words: Vec<String>,
    options: SelfStreamingOptions,
}

impl<B: SelfStreamingBackend> SelfStreamingProcessor<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            audio: Vec::new(),
            offset: 0.0,
            committed_n: 0,
            last_words: Vec::new(),
            options: SelfStreamingOptions::balanced(),
        }
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    fn tokens(&self, words: &[String], lo: usize, hi: usize, total: usize) -> Vec<AsrToken> {
        let dur = self.audio.len() as f64 / SR;
        let total = total.max(1) as f64;
        (lo..hi)
            .map(|j| {
                let s = self.offset + (j as f64 / total) * dur;
                let e = self.offset + ((j as f64 + 1.0) / total) * dur;
                let text = if j > 0 {
                    format!(" {}", words[j])
                } else {
                    words[j].clone()
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

    /// 버퍼 끝 SILENCE_TAIL_SEC 구간이 무음인지.
    fn tail_is_silent(&self) -> bool {
        let n = self.audio.len();
        let tail = self.options.silence_tail_samples;
        if n < tail {
            return false;
        }
        let slice = &self.audio[n - tail..];
        let rms = (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt();
        rms < self.options.silence_rms
    }

    fn has_enough_signal(&self) -> bool {
        let frame = self.options.energy_frame_samples.max(1);
        let mut active = 0usize;
        for slice in self.audio.chunks(frame) {
            if slice.len() < frame / 2 {
                continue;
            }
            let rms = (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt();
            if rms >= self.options.active_rms {
                active += slice.len();
                if active >= self.options.min_active_samples {
                    return true;
                }
            }
        }
        false
    }

    fn clear_window(&mut self) {
        self.offset += self.audio.len() as f64 / SR;
        self.audio.clear();
        self.committed_n = 0;
        self.last_words.clear();
    }

    fn step(&mut self) -> Result<Vec<AsrToken>, AsrError> {
        if self.audio.len() < (SR as usize) / 2 {
            return Ok(vec![]);
        }
        if !self.has_enough_signal() {
            let n = self.audio.len();
            if n >= self.options.max_window_samples
                || (n >= self.options.target_window_samples && self.tail_is_silent())
            {
                self.clear_window();
            }
            return Ok(vec![]);
        }
        // 프롬프트는 넘기지 않는다("") — whisper init_prompt 는 환각/중복을 유발.
        let text = self.backend.transcribe_full(&self.audio, "")?;
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        let mut lcp = 0usize;
        while lcp < words.len() && lcp < self.last_words.len() && words[lcp] == self.last_words[lcp]
        {
            lcp += 1;
        }
        let mut committed = Vec::new();
        if lcp > self.committed_n {
            committed = self.tokens(&words, self.committed_n, lcp, words.len());
            self.committed_n = lcp;
        }
        self.last_words = words;

        // 윈도우 리셋: (목표 길이 + 최근 무음) 또는 강제 상한. 무음 경계에서 끊어
        // 남은 단어를 즉시 확정(지연↓)하고 단어 중간 잘림을 줄인다.
        let n = self.audio.len();
        if n >= self.options.max_window_samples
            || (n >= self.options.target_window_samples && self.tail_is_silent())
        {
            let total = self.last_words.len();
            if total > self.committed_n {
                let mut tail =
                    self.tokens(&self.last_words.clone(), self.committed_n, total, total);
                committed.append(&mut tail);
            }
            self.clear_window();
        }
        Ok(committed)
    }

    fn finish_tokens(&mut self) -> Vec<AsrToken> {
        let total = self.last_words.len();
        if total > self.committed_n {
            self.tokens(&self.last_words.clone(), self.committed_n, total, total)
        } else {
            vec![]
        }
    }
}

#[async_trait]
impl<B: SelfStreamingBackend> StreamingAsrBackend for SelfStreamingProcessor<B> {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        self.options = SelfStreamingOptions::from_config(cfg);
        self.backend.configure(cfg)
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        self.backend.warmup()
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64) {
        if self.audio.is_empty() {
            self.offset = (end_time - pcm.len() as f64 / SR).max(0.0);
        }
        self.audio.extend_from_slice(pcm);
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        if is_last {
            let mut committed = self.step()?;
            committed.extend(self.finish_tokens());
            Ok(committed)
        } else {
            self.step()
        }
    }

    fn get_buffer(&self) -> String {
        self.last_words[self.committed_n.min(self.last_words.len())..].join(" ")
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.backend.set_language(lang);
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps {
            provides_word_timestamps: false,
            provides_probability: false,
            self_streaming: true,
            tokenizer_id: "self_stream",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBackend;

    impl SelfStreamingBackend for MockBackend {
        fn configure(&mut self, _cfg: &AsrConfig) -> Result<(), AsrError> {
            Ok(())
        }

        fn transcribe_full(&mut self, samples: &[f32], _prompt: &str) -> Result<String, AsrError> {
            if samples.len() >= 16_000 {
                Ok("hello world".into())
            } else {
                Ok("hello".into())
            }
        }

        fn set_language(&mut self, _lang: Option<String>) {}
    }

    #[tokio::test]
    async fn final_iter_transcribes_unprocessed_audio() {
        let mut proc = SelfStreamingProcessor::new(MockBackend);
        proc.configure(&AsrConfig::default()).await.unwrap();
        proc.insert_audio_chunk(&vec![0.1; 16_000], 1.0);

        let toks = proc.process_iter(true).await.unwrap();
        let text = toks.iter().map(|t| t.text.as_str()).collect::<String>();

        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn token_times_use_stream_end_time() {
        let mut proc = SelfStreamingProcessor::new(MockBackend);
        proc.configure(&AsrConfig::default()).await.unwrap();
        proc.insert_audio_chunk(&vec![0.1; 16_000], 6.0);

        let toks = proc.process_iter(true).await.unwrap();

        assert!((toks.first().unwrap().start - 5.0).abs() < 0.001);
        assert!((toks.last().unwrap().end - 6.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn silent_iter_skips_backend_call() {
        struct PanicBackend;

        impl SelfStreamingBackend for PanicBackend {
            fn configure(&mut self, _cfg: &AsrConfig) -> Result<(), AsrError> {
                Ok(())
            }

            fn transcribe_full(
                &mut self,
                _samples: &[f32],
                _prompt: &str,
            ) -> Result<String, AsrError> {
                panic!("silent audio should not invoke backend")
            }

            fn set_language(&mut self, _lang: Option<String>) {}
        }

        let mut proc = SelfStreamingProcessor::new(PanicBackend);
        proc.configure(&AsrConfig::default()).await.unwrap();
        proc.insert_audio_chunk(&vec![0.0; 16_000], 1.0);

        let toks = proc.process_iter(false).await.unwrap();

        assert!(toks.is_empty());
    }
}
