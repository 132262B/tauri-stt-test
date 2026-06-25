//! VAD(음성 활동 감지) 추상화 (docs/02-architecture.md, Seed C4 — whisper-live-kit Silero 참고).
//! 구현은 stt-diar(sherpa-onnx Silero). 무음 윈도우의 ASR 추론을 건너뛰어 연산을 절약.

pub trait Vad: Send {
    /// 16kHz mono 샘플을 받아 현재 음성 활동 여부를 반환.
    fn is_speech(&mut self, samples: &[f32]) -> bool;
}
