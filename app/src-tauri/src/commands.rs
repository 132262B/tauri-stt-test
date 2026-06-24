//! 프론트→Rust command (docs/02-architecture.md E.2).
//!
//! P1 커밋4: IPC 헬스체크 `ping` + 마이크 캡처 시작/정지(콘솔 RMS 확인용).
//! start_session/select_backend/export_transcript 등은 이후 추가한다.

use tauri::State;

use crate::app_state::AppState;

/// 프론트가 백엔드 IPC 연결을 확인하는 헬스체크.
#[tauri::command]
pub fn ping() -> &'static str {
    "pong"
}

/// 마이크 캡처 시작 — 16kHz mono 로 리샘플해 mpsc 로 흘리고, 콘솔에 RMS 를 찍는다.
/// 데스크톱 전용(iOS 마이크는 P3).
#[tauri::command]
pub fn start_capture(state: State<AppState>) -> Result<(), String> {
    #[cfg(desktop)]
    {
        let mut guard = state.capture.lock().map_err(|_| "상태 잠금 실패")?;
        if guard.is_some() {
            return Err("이미 캡처 중".into());
        }
        let (tx, rx) = std::sync::mpsc::channel::<crate::capture::AudioFrame>();
        // 커밋4: 소비자는 프레임을 받아 카운트만(파이프라인 연결은 커밋9). RMS 는 콜백에서 로깅.
        std::thread::spawn(move || {
            let mut n: u64 = 0;
            while let Ok(_frame) = rx.recv() {
                n += 1;
                if n % 50 == 0 {
                    eprintln!("[capture] 누적 {n} 프레임 수신");
                }
            }
            eprintln!("[capture] 소비자 종료(누적 {n} 프레임)");
        });
        let control = crate::capture::mic_cpal::start_mic(tx)?;
        *guard = Some(control);
        Ok(())
    }
    #[cfg(not(desktop))]
    {
        let _ = state;
        Err("이 플랫폼의 마이크 캡처는 아직 미구현(P3)".into())
    }
}

/// 마이크 캡처 정지.
#[tauri::command]
pub fn stop_capture(state: State<AppState>) -> Result<(), String> {
    let mut guard = state.capture.lock().map_err(|_| "상태 잠금 실패")?;
    if let Some(control) = guard.take() {
        control.stop();
    }
    Ok(())
}
