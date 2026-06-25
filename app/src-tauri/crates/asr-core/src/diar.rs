//! 화자 분리 추상화 (docs/02-architecture.md I). 구현은 diar-pyannote(sherpa-onnx pyannote).
//!
//! 오프라인 세그먼트 방식: 누적 오디오 전체를 받아 화자별 시간 구간을 반환한다.
//! (온라인 per-segment 방식은 과분할이 심해 폐기 → pyannote segmentation + 글로벌 클러스터링)

/// 화자별 시간 구간(초, 절대시각 기준은 호출측이 offset 보정).
#[derive(Debug, Clone, Copy)]
pub struct DiarSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: u32,
}

/// 16kHz mono 오디오 전체 → 화자 세그먼트 목록.
pub trait Diarizer: Send {
    fn diarize(&mut self, samples: &[f32]) -> Vec<DiarSegment>;
}
