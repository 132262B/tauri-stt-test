//! SenseVoice 다국어 ASR 백엔드 (sherpa-onnx, Rust). 한·영·일·중·월 지원.
//! 발화 단위 모델이라 SelfStreamingProcessor 로 감싸 스트리밍 인터페이스에 맞춘다. Python 0.

use std::path::Path;

use asr_core::asr::{
    AsrConfig, AsrError, SelfStreamingBackend, SelfStreamingProcessor, StreamingAsrBackend,
};
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};

/// SenseVoice 출력의 `<|en|><|NEUTRAL|>…` 같은 메타 태그 제거.
fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars().peekable();
    let mut in_tag = false;
    while let Some(c) = it.next() {
        if !in_tag && c == '<' && it.peek() == Some(&'|') {
            in_tag = true;
            it.next();
            continue;
        }
        if in_tag {
            if c == '|' && it.peek() == Some(&'>') {
                in_tag = false;
                it.next();
            }
            continue;
        }
        out.push(c);
    }
    out.trim().to_string()
}

pub struct SenseVoiceBackend {
    rec: SenseVoiceRecognizer,
}

impl SenseVoiceBackend {
    /// model_dir 안에 model.onnx + tokens.txt 필요. language: None=auto / "ko" / "en" 등.
    pub fn new(model_dir: impl AsRef<Path>, language: Option<String>) -> Result<Self, String> {
        let dir = model_dir.as_ref();
        let config = SenseVoiceConfig {
            model: dir.join("model.onnx").to_string_lossy().into_owned(),
            tokens: dir.join("tokens.txt").to_string_lossy().into_owned(),
            language: language.unwrap_or_else(|| "auto".into()),
            use_itn: true,
            ..Default::default()
        };
        let rec =
            SenseVoiceRecognizer::new(config).map_err(|e| format!("SenseVoice 적재 실패: {e}"))?;
        Ok(Self { rec })
    }
}

impl SelfStreamingBackend for SenseVoiceBackend {
    fn configure(&mut self, _cfg: &AsrConfig) -> Result<(), AsrError> {
        Ok(()) // 모델/언어는 new()에서 적재됨
    }

    fn transcribe_full(&mut self, samples: &[f32], _prompt: &str) -> Result<String, AsrError> {
        let r = self.rec.transcribe(16000, samples);
        Ok(strip_tags(&r.text))
    }

    fn set_language(&mut self, _lang: Option<String>) {
        // SenseVoice 언어는 생성 시 고정(변경하려면 재생성). no-op.
    }
}

/// 모델 디렉터리로부터 StreamingAsrBackend(SelfStreamingProcessor 래핑)를 만든다.
pub fn streaming_backend(
    model_dir: impl AsRef<Path>,
    language: Option<String>,
) -> Result<Box<dyn StreamingAsrBackend>, String> {
    let backend = SenseVoiceBackend::new(model_dir, language)?;
    Ok(Box::new(SelfStreamingProcessor::new(backend)))
}
