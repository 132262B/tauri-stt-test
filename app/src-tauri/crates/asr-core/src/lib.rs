//! asr-core — 온디바이스 회의 전사 코어.
//!
//! 플랫폼/tauri 무의존 순수 Rust. 외부와는 입력 `mpsc<AudioChunk>` /
//! 출력 `mpsc<TranscriptSnapshot>` 채널로만 접한다 (docs/02-architecture.md A.2).

pub mod asr;
pub mod diar;
pub mod eval;
pub mod metrics;
pub mod output;
pub mod pipeline;
pub mod vad;
