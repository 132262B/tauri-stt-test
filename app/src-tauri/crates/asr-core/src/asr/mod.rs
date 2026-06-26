//! 교체형 ASR 백엔드 추상화 (docs/02-architecture.md D).

mod backend;
pub mod self_stream;
mod token;
pub mod wordtime;

pub use backend::{AsrConfig, AsrProfile, BackendCaps, StreamingAsrBackend};
pub use self_stream::{SelfStreamingBackend, SelfStreamingProcessor};
pub use token::{AsrError, AsrToken};
