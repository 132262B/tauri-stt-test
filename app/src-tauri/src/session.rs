//! 라이브 전사 세션 (docs/02-architecture.md B, P1 커밋9).
//!
//! 마이크 캡처(std mpsc AudioFrame) → 브리지 → stt-core run_session(사이드카 ASR) →
//! TranscriptSnapshot → `transcript_update` 이벤트 emit. 정지는 capture.stop() 한 번으로
//! 채널 연쇄가 닫히며 드라이버가 최종 flush 후 종료한다.

use crate::capture::CaptureControl;

/// 실행 중인 세션 핸들. stop() 시 마이크 정지 → 채널 연쇄 종료.
pub struct SessionHandle {
    capture: CaptureControl,
}

impl SessionHandle {
    pub fn stop(self) {
        self.capture.stop();
    }
}

/// 세션을 시작한다(데스크톱). 마이크→사이드카 전사→emit 파이프라인을 띄운다.
#[cfg(desktop)]
pub fn start(app: tauri::AppHandle) -> Result<SessionHandle, String> {
    use std::path::PathBuf;
    use std::sync::mpsc as std_mpsc;

    use tauri::Emitter;
    use tokio::sync::mpsc;

    use stt_asr_sidecar::{SidecarBackend, SidecarSpawn};
    use stt_core::asr::AsrConfig;
    use stt_core::output::TranscriptSnapshot;
    use stt_core::pipeline::{run_session, AudioChunk};

    use crate::capture::{mic_cpal, AudioFrame};

    // 캡처(std mpsc) → 브리지 → run_session(tokio mpsc) → emit
    let (af_tx, af_rx) = std_mpsc::channel::<AudioFrame>();
    let (pcm_tx, pcm_rx) = mpsc::channel::<AudioChunk>(512);
    let (snap_tx, mut snap_rx) = mpsc::channel::<TranscriptSnapshot>(64);

    // 브리지 스레드: AudioFrame → AudioChunk (blocking_send = 백프레셔)
    std::thread::spawn(move || {
        while let Ok(frame) = af_rx.recv() {
            if pcm_tx
                .blocking_send(AudioChunk {
                    samples: frame.samples,
                    t_end: frame.t_end,
                })
                .is_err()
            {
                break;
            }
        }
    });

    // 세션 드라이버(사이드카 spawn + LocalAgreement)
    let sidecar_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sidecar");
    let spawn = SidecarSpawn::dev_venv(sidecar_dir);
    tauri::async_runtime::spawn(async move {
        let backend = SidecarBackend::new(spawn);
        if let Err(e) = run_session(Box::new(backend), AsrConfig::default(), pcm_rx, snap_tx).await {
            eprintln!("[session] run_session 오류: {e}");
        }
    });

    // 스냅샷 → 프론트 emit
    tauri::async_runtime::spawn(async move {
        while let Some(snap) = snap_rx.recv().await {
            let _ = app.emit("transcript_update", &snap);
        }
        let _ = app.emit("transcript_done", ());
    });

    // 마이크 시작(브리지로 송신)
    let capture = mic_cpal::start_mic(af_tx)?;
    Ok(SessionHandle { capture })
}
