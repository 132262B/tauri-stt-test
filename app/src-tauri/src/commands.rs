//! 프론트→Rust command (docs/02-architecture.md E.2).
//!
//! P0 단계에서는 IPC 연결 확인용 `ping` 만 둔다.
//! start_session/stop_session/select_backend/export_transcript 등은 P1 이후 추가한다.

/// 프론트가 백엔드 IPC 연결을 확인하는 헬스체크.
#[tauri::command]
pub fn ping() -> &'static str {
    "pong"
}
