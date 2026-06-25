//! 온라인 화자 분리 추상화 (docs/02-architecture.md I). 구현은 stt-diar(sherpa-onnx).

/// 확정 세그먼트(16kHz mono)를 화자 트랙 id 로 매핑. None=미상.
pub trait Diarizer: Send {
    fn assign(&mut self, samples: &[f32]) -> Option<u32>;
}
