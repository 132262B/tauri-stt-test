//! 교체형 ASR 백엔드 추상화 (docs/02-architecture.md D).

mod backend;
mod token;

pub use backend::{AsrConfig, BackendCaps, StreamingAsrBackend};
pub use token::{AsrError, AsrToken};
