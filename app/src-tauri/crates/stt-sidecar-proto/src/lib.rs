//! Rust↔Python 사이드카 통신 계약 (docs/02-architecture.md C.2).
//!
//! 제어(Rust→py)=stdin NDJSON, 결과(py→Rust)=stdout NDJSON, PCM=UDS 프레임.
//! Python 측 `stt_mlx` 가 동일 필드를 미러한다.

use serde::{Deserialize, Serialize};

/// 제어 메시지 (Rust → 사이드카 stdin, NDJSON 1줄).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Control {
    Configure {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        lang: Option<String>,
        uds_path: String,
        trimming_sec: f32,
    },
    ProcessIter,
    SetLanguage {
        lang: Option<String>,
    },
    ChangeSpeaker {
        at: f64,
    },
    Warmup,
    Finish,
}

/// 결과 토큰(사이드카 → Rust).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub probability: Option<f32>,
    /// 화자 트랙 id(온라인 diarizer). None=미상.
    #[serde(default)]
    pub speaker: Option<u32>,
}

/// 결과 메시지 (사이드카 stdout → Rust, NDJSON 1줄).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Ready {
        backend: String,
        model: String,
        sr: u32,
    },
    Tokens {
        committed: Vec<Token>,
        #[serde(default)]
        buffer: String,
        #[serde(default)]
        upto: f64,
        #[serde(default)]
        is_final: bool,
    },
    Warmed,
    Bye,
    Error {
        code: String,
        msg: String,
    },
}

impl Control {
    /// NDJSON 1줄(개행 포함)로 직렬화.
    pub fn to_ndjson(&self) -> serde_json::Result<String> {
        Ok(format!("{}\n", serde_json::to_string(self)?))
    }
}

/// PCM 프레임 인코딩: `u32 LE n_samples ‖ f32 LE × n ‖ f64 LE t_end`.
pub fn encode_pcm_frame(samples: &[f32], t_end: f64) -> Vec<u8> {
    let n = samples.len();
    let mut buf = Vec::with_capacity(4 + n * 4 + 8);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf.extend_from_slice(&t_end.to_le_bytes());
    buf
}
