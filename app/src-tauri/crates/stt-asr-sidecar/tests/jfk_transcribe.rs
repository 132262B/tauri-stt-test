//! 헤드리스 통합 테스트: JFK WAV → SidecarBackend → run_session → 전사 검증.
//!
//! 프로젝트 로컬 venv(.venv) + 캐시된 모델이 필요하므로 `#[ignore]`.
//! 실행: `cargo test -p stt-asr-sidecar --test jfk_transcribe -- --ignored --nocapture`

use std::path::PathBuf;

use stt_asr_sidecar::{SidecarBackend, SidecarSpawn};
use stt_core::asr::AsrConfig;
use stt_core::metrics::SessionMetrics;
use stt_core::output::TranscriptSnapshot;
use stt_core::pipeline::{run_session, AudioChunk};
use tokio::sync::mpsc;

fn sidecar_dir() -> PathBuf {
    // crates/stt-asr-sidecar → app/src-tauri/sidecar
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../sidecar")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "venv+모델 필요(느림). --ignored 로 명시 실행"]
async fn jfk_streaming_transcription() {
    let dir = sidecar_dir();
    let wav = dir.join("test-data/jfk.wav");
    assert!(wav.exists(), "테스트 오디오 없음: {wav:?} (sidecar/README 참고)");

    // WAV(16k mono i16) → f32
    let mut reader = hound::WavReader::open(&wav).expect("wav open");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "16kHz 가정");
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let backend = SidecarBackend::new(SidecarSpawn::dev_venv(&dir));
    let (pcm_tx, pcm_rx) = mpsc::channel::<AudioChunk>(64);
    let (snap_tx, mut snap_rx) = mpsc::channel::<TranscriptSnapshot>(64);

    let session = tokio::spawn(async move {
        run_session(
            Box::new(backend),
            AsrConfig::default(),
            pcm_rx,
            snap_tx,
            SessionMetrics::default(),
        )
        .await
    });

    // 1초(16000 샘플) 청크 스트리밍
    let sr = 16_000usize;
    let mut t = 0.0_f64;
    for chunk in samples.chunks(sr) {
        t += chunk.len() as f64 / sr as f64;
        pcm_tx
            .send(AudioChunk { samples: chunk.to_vec(), t_end: t })
            .await
            .expect("send pcm");
    }
    drop(pcm_tx); // 캡처 종료 → run_session 최종 flush

    let mut last = TranscriptSnapshot::default();
    while let Some(s) = snap_rx.recv().await {
        eprintln!("[snap] committed={:?} | buffer={:?}", s.committed_text, s.buffer);
        last = s;
    }
    session.await.expect("join").expect("run_session ok");

    let text = last.committed_text.to_lowercase();
    eprintln!("=== FINAL: {}", last.committed_text);
    assert!(text.contains("country"), "전사에 'country' 가 없음: {text:?}");
    assert!(text.contains("americans"), "전사에 'americans' 가 없음: {text:?}");
}
