//! LocalAgreement-2 스트리밍 정책 (Rust 포팅) — docs/02-architecture.md D.3.
//!
//! whisper-live-kit online_asr.py 의 HypothesisBuffer/OnlineASRProcessor 를 Rust 로 이식.
//! 연속 2회 추론의 최장 공통 접두사만 확정. segment 트리밍은 마지막 확정 시각 기준으로 단순화.

use super::backend::WhisperLikeBackend;
use super::token::{AsrError, AsrToken};

const SR: f64 = 16_000.0;

fn with_offset(mut t: AsrToken, offset: f64) -> AsrToken {
    t.start += offset;
    t.end += offset;
    t
}

/// 확정/미확정/신규 토큰 버퍼.
#[derive(Default)]
pub struct HypothesisBuffer {
    committed_in_buffer: Vec<AsrToken>,
    pub buffer: Vec<AsrToken>,
    new: Vec<AsrToken>,
    last_committed_time: f64,
}

impl HypothesisBuffer {
    fn insert(&mut self, new_tokens: Vec<AsrToken>, offset: f64) {
        let lct = self.last_committed_time;
        self.new = new_tokens
            .into_iter()
            .map(|t| with_offset(t, offset))
            .filter(|t| t.start > lct - 0.1)
            .collect();

        if let Some(first) = self.new.first() {
            if (first.start - self.last_committed_time).abs() < 1.0 && !self.committed_in_buffer.is_empty() {
                let cl = self.committed_in_buffer.len();
                let nl = self.new.len();
                let max_ngram = cl.min(nl).min(5);
                for i in 1..=max_ngram {
                    let c: String = self.committed_in_buffer[cl - i..]
                        .iter()
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let n: String = self.new[..i]
                        .iter()
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if c == n {
                        self.new.drain(0..i);
                        break;
                    }
                }
            }
        }
    }

    fn flush(&mut self) -> Vec<AsrToken> {
        let mut committed = Vec::new();
        while !self.new.is_empty() {
            if self.buffer.is_empty() {
                break;
            }
            if self.new[0].text == self.buffer[0].text {
                let t = self.new.remove(0);
                self.last_committed_time = t.end;
                committed.push(t);
                self.buffer.remove(0);
            } else {
                break;
            }
        }
        self.buffer = std::mem::take(&mut self.new);
        self.committed_in_buffer.extend(committed.iter().cloned());
        committed
    }

    fn pop_committed(&mut self, time: f64) {
        while !self.committed_in_buffer.is_empty() && self.committed_in_buffer[0].end <= time {
            self.committed_in_buffer.remove(0);
        }
    }
}

/// 스트리밍 오디오를 누적해 주기적으로 백엔드를 호출하고 LocalAgreement 로 확정/트림.
pub struct OnlineAsrProcessor<B: WhisperLikeBackend> {
    backend: B,
    audio_buffer: Vec<f32>,
    buffer_time_offset: f64,
    hyp: HypothesisBuffer,
    committed: Vec<AsrToken>,
    trimming_sec: f64,
}

impl<B: WhisperLikeBackend> OnlineAsrProcessor<B> {
    pub fn new(backend: B, trimming_sec: f32) -> Self {
        Self {
            backend,
            audio_buffer: Vec::new(),
            buffer_time_offset: 0.0,
            hyp: HypothesisBuffer::default(),
            committed: Vec::new(),
            trimming_sec: trimming_sec as f64,
        }
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn insert_audio_chunk(&mut self, pcm: &[f32]) {
        self.audio_buffer.extend_from_slice(pcm);
    }

    pub fn audio_end_time(&self) -> f64 {
        self.buffer_time_offset + self.audio_buffer.len() as f64 / SR
    }

    fn prompt(&self) -> String {
        let mut k = self.committed.len();
        while k > 0 && self.committed[k - 1].end > self.buffer_time_offset {
            k -= 1;
        }
        let mut words: Vec<&str> = self.committed[..k].iter().map(|t| t.text.as_str()).collect();
        let mut picked: Vec<&str> = Vec::new();
        let mut len_count = 0usize;
        while let Some(w) = words.pop() {
            if len_count >= 200 {
                break;
            }
            len_count += w.len() + 1;
            picked.push(w);
        }
        picked.reverse();
        picked.join(self.backend.sep())
    }

    pub fn process_iter(&mut self) -> Result<Vec<AsrToken>, AsrError> {
        let prompt = self.prompt();
        let tokens = self.backend.transcribe(&self.audio_buffer, &prompt)?;
        self.hyp.insert(tokens, self.buffer_time_offset);
        let committed = self.hyp.flush();
        self.committed.extend(committed.iter().cloned());

        let dur = self.audio_buffer.len() as f64 / SR;
        if dur > self.trimming_sec {
            if let Some(last) = self.committed.last() {
                let t = last.end;
                self.chunk_at(t);
            } else {
                self.chunk_at(self.buffer_time_offset + dur / 2.0);
            }
        }
        Ok(committed)
    }

    fn chunk_at(&mut self, time: f64) {
        self.hyp.pop_committed(time);
        let cut = (((time - self.buffer_time_offset) * SR) as i64).max(0) as usize;
        if cut >= self.audio_buffer.len() {
            self.audio_buffer.clear();
        } else {
            self.audio_buffer.drain(0..cut);
        }
        self.buffer_time_offset = time;
    }

    pub fn get_buffer(&self) -> String {
        self.hyp
            .buffer
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(self.backend.sep())
    }

    pub fn finish(&mut self) -> Vec<AsrToken> {
        std::mem::take(&mut self.hyp.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(start: f64, end: f64, text: &str) -> AsrToken {
        AsrToken {
            start,
            end,
            text: text.into(),
            probability: None,
            detected_language: None,
            speaker: None,
        }
    }

    /// 연속 2회 동일 토큰만 확정되는지(LocalAgreement-2 핵심).
    #[test]
    fn local_agreement_commits_only_agreed_prefix() {
        let mut h = HypothesisBuffer::default();
        // 1회차: "the cat" — buffer 비어있으므로 commit 0, buffer=[the,cat]
        h.insert(vec![tk(0.0, 0.3, "the"), tk(0.3, 0.6, "cat")], 0.0);
        let c1 = h.flush();
        assert!(c1.is_empty());
        // 2회차: "the cat sat" — the,cat 가 buffer 와 일치 → 확정, sat 는 buffer 로
        h.insert(vec![tk(0.0, 0.3, "the"), tk(0.3, 0.6, "cat"), tk(0.6, 0.9, "sat")], 0.0);
        let c2 = h.flush();
        let text: Vec<&str> = c2.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(text, vec!["the", "cat"], "확정은 합의된 the cat 만");
        // 3회차: "the cat sat down" — sat 확정, down buffer
        h.insert(
            vec![
                tk(0.0, 0.3, "the"),
                tk(0.3, 0.6, "cat"),
                tk(0.6, 0.9, "sat"),
                tk(0.9, 1.2, "down"),
            ],
            0.0,
        );
        let c3 = h.flush();
        let text3: Vec<&str> = c3.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(text3, vec!["sat"]);
    }
}
