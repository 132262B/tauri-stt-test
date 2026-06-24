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
/// model: MLX Whisper HF repo id(None=기본 turbo).
#[tauri::command]
pub fn start_session(
    app: tauri::AppHandle,
    state: State<AppState>,
    model: Option<String>,
) -> Result<(), String> {
    #[cfg(desktop)]
    {
        let mut guard = state.session.lock().map_err(|_| "상태 잠금 실패")?;
        if guard.is_some() {
            return Err("이미 세션 진행 중".into());
        }
        let handle = crate::session::start(app, state.transcript.clone(), model)?;
        *guard = Some(handle);
        Ok(())
    }
    #[cfg(not(desktop))]
    {
        let _ = (app, state, model);
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

/// 현재 세션 전사를 파일로 내보낸다(txt/srt/json). path 는 프론트 저장 대화상자에서.
#[tauri::command]
pub fn export_transcript(state: State<AppState>, path: String, format: String) -> Result<String, String> {
    let tokens = state
        .transcript
        .lock()
        .map_err(|_| "상태 잠금 실패")?
        .clone();
    if tokens.is_empty() {
        return Err("내보낼 전사 내용이 없습니다".into());
    }
    let content = match format.as_str() {
        "srt" => crate::export::to_srt(&tokens),
        "json" => crate::export::to_json(&tokens),
        _ => crate::export::to_txt(&tokens),
    };
    std::fs::write(&path, content).map_err(|e| format!("저장 실패: {e}"))?;
    Ok(path)
}
