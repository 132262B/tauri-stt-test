//! 앱 전역 상태 (docs/02-architecture.md A.1).
//!
//! 이후 ModelManager, 선택된 ASR 백엔드 등을 보유한다.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use asr_core::output::CommittedToken;

use crate::session::SessionHandle;

/// Tauri `.manage()`로 주입되는 앱 전역 상태.
#[derive(Default)]
pub struct AppState {
    /// 실행 중인 전사 세션(P1 커밋9). stop 시 take 하여 정지.
    pub session: Mutex<Option<SessionHandle>>,
    /// 현재 세션의 확정 토큰 누적(내보내기용, P1 커밋13).
    pub transcript: Arc<Mutex<Vec<CommittedToken>>>,
    /// 초기화 요청 플래그. 세트되면 드라이버가 누적 라인/버퍼를 비운다(전사 중에도 동작).
    pub reset: Arc<AtomicBool>,
}
