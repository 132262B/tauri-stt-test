use serde::Serialize;

/// 확정 토큰(절대시각). 내보내기(srt/json)·타임스탬프 보존용.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommittedToken {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

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
    /// 이번 iter 에 새로 확정된 토큰(누적용 — 내보내기에서 사용).
    pub new_committed: Vec<CommittedToken>,
}

/// 자원/성능 스냅샷 (docs/02-architecture.md H). 1초 주기 emit.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricsSnapshot {
    /// 앱+사이드카 합산 CPU 사용률(%).
    pub cpu_pct: f32,
    /// 앱 프로세스 RSS(MB).
    pub rss_mb: f32,
    /// 사이드카(자식 프로세스) RSS 합산(MB) — MLX 모델 메모리 대부분이 여기.
    pub sidecar_rss_mb: f32,
    /// real-time factor (추론시간/오디오길이). <1 이면 실시간 여유.
    pub rtf: f32,
    pub latency_ms_p50: f32,
    pub latency_ms_p95: f32,
    pub backend: String,
    pub model: String,
}
