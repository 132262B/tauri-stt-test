use async_trait::async_trait;

use super::token::{AsrError, AsrToken};

/// ASR 백엔드 설정. language=None 은 자동감지(한·영 코드스위칭).
#[derive(Clone, Debug)]
pub struct AsrConfig {
    pub model_id: String,
    pub language: Option<String>,
    pub trimming_sec: f32,
    pub profile: AsrProfile,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            model_id: "mlx-community/whisper-large-v3-turbo".to_string(),
            language: None,
            trimming_sec: 15.0,
            profile: AsrProfile::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AsrProfile {
    /// Pick the safest profile from model_id.
    #[default]
    Auto,
    /// Preserve the historical decoding behavior.
    Balanced,
    /// Low-latency Q5_0 turbo profile for live captions.
    RealtimeQ5,
}

impl AsrConfig {
    pub fn effective_profile(&self) -> AsrProfile {
        match self.profile {
            AsrProfile::Auto if is_q5_turbo(&self.model_id) => AsrProfile::RealtimeQ5,
            AsrProfile::Auto => AsrProfile::Balanced,
            p => p,
        }
    }
}

fn is_q5_turbo(model_id: &str) -> bool {
    let m = model_id.to_ascii_lowercase();
    m.contains("large-v3-turbo") && m.contains("q5_0")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_uses_realtime_profile_for_q5_turbo() {
        let cfg = AsrConfig {
            model_id: "ggml-large-v3-turbo-q5_0".into(),
            ..AsrConfig::default()
        };
        assert_eq!(cfg.effective_profile(), AsrProfile::RealtimeQ5);
    }

    #[test]
    fn explicit_balanced_overrides_q5_turbo_detection() {
        let cfg = AsrConfig {
            model_id: "ggml-large-v3-turbo-q5_0".into(),
            profile: AsrProfile::Balanced,
            ..AsrConfig::default()
        };
        assert_eq!(cfg.effective_profile(), AsrProfile::Balanced);
    }
}

/// 백엔드 이질성을 정책/파이프라인이 런타임에 분기(docs/02-architecture.md D.2).
#[derive(Clone, Copy, Debug)]
pub struct BackendCaps {
    /// false 면 LocalAgreement 불가 → segment-only 폴백.
    pub provides_word_timestamps: bool,
    /// 조기확정(prob>0.95) 가능 여부.
    pub provides_probability: bool,
    /// true = Voxtral/Qwen(자체 버퍼 재전사), false = Whisper류.
    pub self_streaming: bool,
    /// hot-swap 시 토큰 체계 차이 식별.
    pub tokenizer_id: &'static str,
}

/// 라이브 스트리밍 ASR 백엔드. 사이드카가 첫 구현, 네이티브(mlx-rs/mlx-swift)가 drop-in.
#[async_trait]
pub trait StreamingAsrBackend: Send {
    /// 모델/언어 설정 + 준비(사이드카 spawn·핸드셰이크 포함).
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError>;
    /// 번들 PCM 1회 예열(선택).
    async fn warmup(&mut self) -> Result<(), AsrError>;
    /// 16kHz mono PCM 청크 공급(논블로킹).
    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64);
    /// 1회 추론 → 확정 토큰. is_last 면 잔여 flush.
    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError>;
    /// 미확정 partial(캐시값).
    fn get_buffer(&self) -> String;
    /// 언어 변경(None=auto).
    fn set_language(&mut self, lang: Option<String>);
    /// 런타임 능력 질의.
    fn caps(&self) -> BackendCaps;
}
