//! 라이브 파이프라인 드라이버 (docs/02-architecture.md B).

mod driver;

pub use driver::{run_session, AudioChunk};
