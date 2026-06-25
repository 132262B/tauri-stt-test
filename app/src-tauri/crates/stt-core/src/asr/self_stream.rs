//! 자체 스트리밍 ASR 래퍼 (발화 단위 모델용) — docs/02-architecture.md D.5.
//!
//! SenseVoice 등 "버퍼 전체 → 텍스트" 모델을 단어 prefix-commit(연속 2회 전사의 최장 공통
//! 접두사 확정)으로 StreamingAsrBackend 인터페이스에 맞춘다. 타임스탬프는 윈도우 비례 근사.

use async_trait::async_trait;

use super::backend::{AsrConfig, BackendCaps, StreamingAsrBackend};
use super::token::{AsrError, AsrToken};

const SR: f64 = 16_000.0;
const WINDOW_SAMPLES: usize = 24 * 16_000; // 24s 윈도우(초과 시 flush)

/// "16kHz mono 버퍼 → 전체 텍스트" 동기 백엔드(SenseVoice 등).
pub trait SelfStreamingBackend: Send {
    fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError>;
    fn transcribe_full(&mut self, samples: &[f32]) -> Result<String, AsrError>;
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

    fn step(&mut self) -> Result<Vec<AsrToken>, AsrError> {
        if self.audio.len() < (SR as usize) / 2 {
            return Ok(vec![]);
        }
        let text = self.backend.transcribe_full(&self.audio)?;
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

        // 윈도우 초과 → 전부 확정 후 초기화
        if self.audio.len() >= WINDOW_SAMPLES {
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
