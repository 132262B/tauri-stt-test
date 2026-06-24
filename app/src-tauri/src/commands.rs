//! 프론트→Rust command (docs/02-architecture.md E.2).
//!
//! P1 커밋9: IPC 헬스체크 `ping` + 전사 세션 시작/정지(마이크→사이드카→transcript_update).
//! select_backend/export_transcript/모델 관리 등은 이후 추가한다.

use tauri::State;

use crate::app_state::AppState;

/// 프론트가 백엔드 IPC 연결을 확인하는 헬스체크.
#[tauri::command]
pub fn ping() -> &'static str {
    "pong"
}

/// 전사 세션 시작 — 마이크 캡처 + 사이드카 MLX Whisper 전사. 데스크톱 전용(iOS는 P3).
#[tauri::command]
pub fn start_session(app: tauri::AppHandle, state: State<AppState>) -> Result<(), String> {
    #[cfg(desktop)]
    {
        let mut guard = state.session.lock().map_err(|_| "상태 잠금 실패")?;
        if guard.is_some() {
            return Err("이미 세션 진행 중".into());
        }
        let handle = crate::session::start(app)?;
        *guard = Some(handle);
        Ok(())
    }
    #[cfg(not(desktop))]
    {
        let _ = (app, state);
        Err("이 플랫폼의 전사는 아직 미지원(P3)".into())
    }
}

/// 전사 세션 정지.
#[tauri::command]
pub fn stop_session(state: State<AppState>) -> Result<(), String> {
    let mut guard = state.session.lock().map_err(|_| "상태 잠금 실패")?;
    if let Some(handle) = guard.take() {
        handle.stop();
    }
    Ok(())
}
