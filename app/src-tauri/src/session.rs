//! 라이브 전사 세션 (docs/02-architecture.md B·H, P1 커밋9·12).
//!
//! 마이크 캡처(std mpsc AudioFrame) → 브리지 → stt-core run_session(사이드카 ASR) →
//! TranscriptSnapshot → `transcript_update` emit. 별도 1초 주기 task 가 CPU/RSS(sysinfo)+
//! RTF/지연을 합쳐 `metrics_update` emit. 정지는 capture.stop()+stop_flag 로 연쇄 종료.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::capture::CaptureControl;

/// 실행 중인 세션 핸들. stop() 시 캡처(들) 정지 + 모니터 task 종료.
pub struct SessionHandle {
    captures: Vec<CaptureControl>,
    stop_flag: Arc<AtomicBool>,
}

impl SessionHandle {
    pub fn stop(self) {
        self.stop_flag.store(false, Ordering::Relaxed);
        for c in self.captures {
            c.stop();
        }
    }
}

/// 세션을 시작한다(데스크톱). 입력(mic/system/both)→전사→emit + 자원 모니터.
/// transcript_log 에 확정 토큰을 누적(내보내기용).
#[cfg(desktop)]
pub fn start(
    app: tauri::AppHandle,
    transcript_log: Arc<std::sync::Mutex<Vec<stt_core::output::CommittedToken>>>,
    model_id: Option<String>,
    lang: Option<String>,
    input: String,
    device: Option<String>,
) -> Result<SessionHandle, String> {
    use std::path::PathBuf;
    use std::sync::mpsc as std_mpsc;
    use std::time::Duration;

    use tauri::Emitter;
    use tokio::sync::mpsc;

    use stt_asr_whisper::WhisperStreamingBackend;
    use stt_core::asr::{AsrConfig, StreamingAsrBackend};
    use stt_core::diar::Diarizer;
    use stt_core::vad::Vad;
    use stt_core::metrics::SessionMetrics;
    use stt_core::output::{MetricsSnapshot, TranscriptSnapshot};
    use stt_core::pipeline::{run_session, AudioChunk};

    use crate::capture::{mic_cpal, AudioFrame};

    // 새 세션이므로 이전 전사 누적 초기화.
    if let Ok(mut g) = transcript_log.lock() {
        g.clear();
    }

    // 기본은 Rust 네이티브 Whisper(ggml). Voxtral/Qwen 만 Python 사이드카.
    let model_id = model_id.unwrap_or_else(|| "ggml-large-v3-turbo".to_string());
    let cfg = AsrConfig {
        model_id: model_id.clone(),
        language: lang,
        ..AsrConfig::default()
    };
    let model = model_id.clone();
    let metrics = SessionMetrics::default();
    let running = Arc::new(AtomicBool::new(true));

    // 캡처(std mpsc) → 브리지 → run_session(tokio mpsc) → emit
    let (af_tx, af_rx) = std_mpsc::channel::<AudioFrame>();
    let (pcm_tx, pcm_rx) = mpsc::channel::<AudioChunk>(512);
    let (snap_tx, mut snap_rx) = mpsc::channel::<TranscriptSnapshot>(64);

    // 브리지 스레드: AudioFrame → AudioChunk (blocking_send = 백프레셔).
    // 입력 레벨(RMS)을 ~10Hz로 audio_level emit — 사용자가 음성 입력을 눈으로 확인.
    let app_level = app.clone();
    std::thread::spawn(move || {
        let mut acc = 0f32;
        let mut cnt = 0usize;
        while let Ok(frame) = af_rx.recv() {
            for &s in &frame.samples {
                acc += s * s;
            }
            cnt += frame.samples.len();
            if cnt >= 1600 {
                let rms = (acc / cnt as f32).sqrt();
                let _ = app_level.emit("audio_level", rms);
                acc = 0.0;
                cnt = 0;
            }
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
        // 종료 시 레벨 0
        let _ = app_level.emit("audio_level", 0.0_f32);
    });

    // 전사는 전부 Rust 네이티브(in-process). Python/Node 프로세스 없음.
    // ggml-* = Whisper(whisper.cpp), sensevoice = SenseVoice(sherpa-onnx 다국어).
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let backend: Box<dyn StreamingAsrBackend> = if model_id == "sensevoice" {
        stt_asr_sense::streaming_backend(crate_dir.join("models/sense"), cfg.language.clone())?
    } else {
        Box::new(WhisperStreamingBackend::new(crate_dir.join("models/ggml")))
    };

    // 온라인 화자 분리(sherpa-onnx, Rust). 모델 자동 다운로드. 실패 시 화자 라벨 없이 진행.
    let diarizer: Option<Box<dyn Diarizer>> =
        match stt_diar::OnlineDiarizer::with_download(crate_dir.join("models/speaker"), 0.5) {
            Ok(d) => Some(Box::new(d)),
            Err(e) => {
                eprintln!("[session] 화자분리 비활성: {e}");
                None
            }
        };

    // VAD 게이트 자리(trait+param 준비). sherpa-rs Silero 는 현재 SIGSEGV 라 보류 —
    // whisper.cpp 가 무음을 자체 처리하므로 None 으로 진행(docs/03-progress.md).
    let vad: Option<Box<dyn Vad>> = None;

    let metrics_for_driver = metrics.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) =
            run_session(backend, cfg, pcm_rx, snap_tx, metrics_for_driver, diarizer, vad).await
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

    // 입력 소스 시작. mic/system/both. 시스템 오디오는 macOS ScreenCaptureKit(C7).
    #[cfg(target_os = "macos")]
    let captures: Vec<CaptureControl> = match input.as_str() {
        "system" => vec![crate::capture::screencapturekit::start_system(af_tx)?],
        "both" => {
            let (mic_tx, mic_rx) = std_mpsc::channel::<AudioFrame>();
            let (sys_tx, sys_rx) = std_mpsc::channel::<AudioFrame>();
            let mic_c = mic_cpal::start_mic(mic_tx, device.clone())?;
            let sys_c = crate::capture::screencapturekit::start_system(sys_tx)?;
            crate::capture::mixer::spawn_mixer(mic_rx, sys_rx, af_tx);
            vec![mic_c, sys_c]
        }
        _ => vec![mic_cpal::start_mic(af_tx, device.clone())?],
    };
    #[cfg(not(target_os = "macos"))]
    let captures: Vec<CaptureControl> = {
        let _ = &input;
        vec![mic_cpal::start_mic(af_tx, device.clone())?]
    };

    Ok(SessionHandle {
        captures,
        stop_flag: running,
    })
}
