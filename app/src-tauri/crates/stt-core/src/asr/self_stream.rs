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

use super::backend::{AsrConfig, BackendCaps, StreamingAsrBackend};
use super::token::{AsrError, AsrToken};

const SR: f64 = 16_000.0;
/// 이 길이를 넘고 최근이 무음이면 윈도우 리셋(깨끗한 경계에서 끊어 남은 단어 빨리 확정).
const TARGET_WINDOW: usize = 12 * 16_000;
/// 무음이 안 와도 이 길이에서는 강제 리셋(틱 비용/버퍼 폭주 방지).
const MAX_WINDOW: usize = 20 * 16_000;
/// 리셋 가능한 "무음" 판정 RMS 임계.
const SILENCE_RMS: f32 = 0.012;
/// 무음 판정에 보는 버퍼 끝부분 길이(초).
const SILENCE_TAIL_SEC: f64 = 0.4;

/// "16kHz mono 버퍼 → 전체 텍스트" 동기 백엔드(SenseVoice/Whisper/Qwen 등).
/// prompt: 직전 확정 텍스트(연속성용). 지원 안 하면 무시 가능.
pub trait SelfStreamingBackend: Send {
    fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError>;
    fn transcribe_full(&mut self, samples: &[f32], prompt: &str) -> Result<String, AsrError>;
    fn set_language(&mut self, lang: Option<String>);
}

pub struct SelfStreamingProcessor<B: SelfStreamingBackend> {
    backend: B,
    audio: Vec<f32>,
    offset: f64,
    committed_n: usize,
    last_words: Vec<String>,
}

impl<B: SelfStreamingBackend> SelfStreamingProcessor<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            audio: Vec::new(),
            offset: 0.0,
            committed_n: 0,
            last_words: Vec::new(),
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
        let tail = (SILENCE_TAIL_SEC * SR) as usize;
        if n < tail {
            return false;
        }
        let slice = &self.audio[n - tail..];
        let rms = (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt();
        rms < SILENCE_RMS
    }

    fn step(&mut self) -> Result<Vec<AsrToken>, AsrError> {
        if self.audio.len() < (SR as usize) / 2 {
            return Ok(vec![]);
        }
        // 프롬프트는 넘기지 않는다("") — whisper init_prompt 는 환각/중복을 유발.
        let text = self.backend.transcribe_full(&self.audio, "")?;
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        let mut lcp = 0usize;
        while lcp < words.len() && lcp < self.last_words.len() && words[lcp] == self.last_words[lcp] {
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
        if n >= MAX_WINDOW || (n >= TARGET_WINDOW && self.tail_is_silent()) {
            let total = self.last_words.len();
            if total > self.committed_n {
                let mut tail = self.tokens(&self.last_words.clone(), self.committed_n, total, total);
                committed.append(&mut tail);
            }
            self.offset += self.audio.len() as f64 / SR;
            self.audio.clear();
            self.committed_n = 0;
            self.last_words.clear();
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
        self.backend.configure(cfg)
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        Ok(())
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], _end_time: f64) {
        self.audio.extend_from_slice(pcm);
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        if is_last {
            Ok(self.finish_tokens())
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
