//! 앱 전역 상태 (docs/02-architecture.md A.1).
//!
//! P0 단계에서는 빈 골격. 이후 SessionHandle, ModelManager, 선택된 ASR 백엔드 등을 보유한다.

/// Tauri `.manage()`로 주입되는 앱 전역 상태.
#[derive(Default)]
pub struct AppState {}
