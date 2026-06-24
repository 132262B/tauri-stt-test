//! 라이브 전사 세션 (docs/02-architecture.md B·H, P1 커밋9·12).
//!
//! 마이크 캡처(std mpsc AudioFrame) → 브리지 → stt-core run_session(사이드카 ASR) →
//! TranscriptSnapshot → `transcript_update` emit. 별도 1초 주기 task 가 CPU/RSS(sysinfo)+
//! RTF/지연을 합쳐 `metrics_update` emit. 정지는 capture.stop()+stop_flag 로 연쇄 종료.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::capture::CaptureControl;

/// 실행 중인 세션 핸들. stop() 시 마이크 정지 + 모니터 task 종료.
pub struct SessionHandle {
    capture: CaptureControl,
    stop_flag: Arc<AtomicBool>,
}

impl SessionHandle {
    pub fn stop(self) {
        self.stop_flag.store(false, Ordering::Relaxed);
        self.capture.stop();
    }
}

/// 세션을 시작한다(데스크톱). 마이크→사이드카 전사→emit + 자원 모니터.
/// transcript_log 에 확정 토큰을 누적(내보내기용).
#[cfg(desktop)]
pub fn start(
    app: tauri::AppHandle,
    transcript_log: Arc<std::sync::Mutex<Vec<stt_core::output::CommittedToken>>>,
    model_id: Option<String>,
) -> Result<SessionHandle, String> {
    use std::path::PathBuf;
    use std::sync::mpsc as std_mpsc;
    use std::time::Duration;

    use tauri::Emitter;
    use tokio::sync::mpsc;

    use stt_asr_sidecar::{SidecarBackend, SidecarSpawn};
    use stt_core::asr::AsrConfig;
    use stt_core::metrics::SessionMetrics;
    use stt_core::output::{MetricsSnapshot, TranscriptSnapshot};
    use stt_core::pipeline::{run_session, AudioChunk};

    use crate::capture::{mic_cpal, AudioFrame};

    // 새 세션이므로 이전 전사 누적 초기화.
    if let Ok(mut g) = transcript_log.lock() {
        g.clear();
    }

    let cfg = AsrConfig {
        model_id: model_id.unwrap_or_else(|| AsrConfig::default().model_id),
        ..AsrConfig::default()
    };
    let model = cfg.model_id.clone();
    let metrics = SessionMetrics::default();
    let running = Arc::new(AtomicBool::new(true));

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
    let metrics_for_driver = metrics.clone();
    tauri::async_runtime::spawn(async move {
        let backend = SidecarBackend::new(spawn);
        if let Err(e) =
            run_session(Box::new(backend), cfg, pcm_rx, snap_tx, metrics_for_driver).await
        {
            eprintln!("[session] run_session 오류: {e}");
        }
    });

    // 스냅샷 → 전사 누적 + 프론트 emit
    let app_tx = app.clone();
    let log_tx = transcript_log.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(snap) = snap_rx.recv().await {
            if !snap.new_committed.is_empty() {
                if let Ok(mut g) = log_tx.lock() {
                    g.extend(snap.new_committed.iter().cloned());
                }
            }
            let _ = app_tx.emit("transcript_update", &snap);
        }
        let _ = app_tx.emit("transcript_done", ());
    });

    // 자원 모니터 task(1초): 앱+사이드카 CPU/RSS + RTF/지연 → metrics_update
    let app_metrics = app.clone();
    let metrics_for_task = metrics.clone();
    let running_for_task = running.clone();
    tauri::async_runtime::spawn(async move {
        let mut sys = sysinfo::System::new_all();
        let me = std::process::id();
        while running_for_task.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(1)).await;
            sys.refresh_all();
            let mut app_rss = 0u64;
            let mut child_rss = 0u64;
            let mut cpu = 0f32;
            for (pid, proc_) in sys.processes() {
                let is_me = pid.as_u32() == me;
                let is_child = proc_.parent().map(|p| p.as_u32()) == Some(me);
                if is_me {
                    app_rss += proc_.memory();
                    cpu += proc_.cpu_usage();
                } else if is_child {
                    child_rss += proc_.memory();
                    cpu += proc_.cpu_usage();
                }
            }
            let mc = metrics_for_task.snapshot();
            let snap = MetricsSnapshot {
                cpu_pct: cpu,
                rss_mb: app_rss as f32 / 1_048_576.0,
                sidecar_rss_mb: child_rss as f32 / 1_048_576.0,
                rtf: mc.rtf,
                latency_ms_p50: mc.latency_p50,
                latency_ms_p95: mc.latency_p95,
                backend: "mlx_whisper".into(),
                model: model.clone(),
            };
            if app_metrics.emit("metrics_update", &snap).is_err() {
                break;
            }
        }
    });

    // 마이크 시작(브리지로 송신)
    let capture = mic_cpal::start_mic(af_tx)?;
    Ok(SessionHandle {
        capture,
        stop_flag: running,
    })
}
