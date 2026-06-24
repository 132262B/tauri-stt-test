use serde::Serialize;

/// 전사 스냅샷(전체 모드). 화자 라벨·diff/relabel·라인 분할은 P1.5/P2에서 확장.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSnapshot {
    /// 확정된 전사 텍스트(누적).
    pub committed_text: String,
    /// 미확정 partial.
    pub buffer: String,
    /// 처리된 오디오 끝 시각(초).
    pub upto: f64,
}
