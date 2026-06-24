//! stt-core — 온디바이스 회의 전사 코어.
//!
//! 플랫폼/tauri 무의존 순수 Rust. 외부와는 입력 `mpsc<PcmFrame>` /
//! 출력 `broadcast<PipelineEvent>` 두 채널로만 접한다.
//!
//! 모듈 구조(예정, docs/02-architecture.md A.1):
//! `audio` / `vad` / `asr` / `diar` / `pipeline` / `output` / `models` / `metrics` / `config`.
//!
//! P0 단계에서는 빈 골격만 둔다.
