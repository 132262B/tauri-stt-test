//! stt-sidecar-proto — Rust↔Python 사이드카 통신 계약.
//!
//! 제어(Rust→py)/결과(py→Rust)는 stdout NDJSON, PCM 전송은 Unix Domain Socket
//! (`u32 LE length ‖ f32 LE PCM ‖ 8B f64 t_end`) — docs/02-architecture.md C.2.
//!
//! P0 단계에서는 빈 골격만 둔다. serde 구조체는 C.2 구현 시 추가한다.
