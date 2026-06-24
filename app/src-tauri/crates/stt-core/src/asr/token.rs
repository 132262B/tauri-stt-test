use serde::{Deserialize, Serialize};

/// 단어 단위 전사 토큰(절대시각). MLX Whisper 는 probability 미제공(None).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AsrToken {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub probability: Option<f32>,
    /// ko/en 코드스위칭(토큰/세그먼트 단위). 현재 사이드카는 미제공.
    #[serde(default)]
    pub detected_language: Option<String>,
    /// 화자 트랙 id(온라인 diarizer). None=미상.
    #[serde(default)]
    pub speaker: Option<u32>,
}

#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("backend not ready")]
    NotReady,
    #[error("sidecar died")]
    BackendDied,
    #[error("model not found offline: {0}")]
    ModelMissing(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("cancelled")]
    Cancelled,
}
