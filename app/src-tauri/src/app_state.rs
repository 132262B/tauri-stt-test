//! 앱 전역 상태 (docs/02-architecture.md A.1).
//!
//! 이후 SessionHandle, ModelManager, 선택된 ASR 백엔드 등을 보유한다.

use std::sync::Mutex;

use crate::capture::CaptureControl;

/// Tauri `.manage()`로 주입되는 앱 전역 상태.
#[derive(Default)]
pub struct AppState {
    /// 실행 중인 마이크 캡처 핸들(P1 커밋4). stop 시 take 하여 정지.
    pub capture: Mutex<Option<CaptureControl>>,
}
