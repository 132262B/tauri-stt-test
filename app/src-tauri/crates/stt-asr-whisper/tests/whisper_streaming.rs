//! 전체 Rust 네이티브 스트리밍 경로 검증: WhisperStreamingBackend + run_session(LocalAgreement). Python 0.
//! 실행: `cargo test -p stt-asr-whisper --test whisper_streaming -- --ignored --nocapture`

use std::path::PathBuf;

use stt_asr_whisper::WhisperStreamingBackend;
use stt_core::asr::AsrConfig;
use stt_core::metrics::SessionMetrics;
use stt_core::output::TranscriptSnapshot;
use stt_core::pipeline::{run_session, AudioChunk};
use tokio::sync::mpsc;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ggml 모델 필요. --ignored 로 실행"]
async fn jfk_rust_native_streaming() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../sidecar");
    let ggml = base.join(".hf-cache/ggml");
    let wav = base.join("test-data/jfk.wav");
    let mut reader = hound::WavReader::open(&wav).expect("wav");
    let audio: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let backend = Box::new(WhisperStreamingBackend::new(ggml));
    let cfg = AsrConfig {
        model_id: "ggml-base".into(),
        language: Some("en".into()),
        trimming_sec: 15.0,
    };
    let (pcm_tx, pcm_rx) = mpsc::channel::<AudioChunk>(64);
    let (snap_tx, mut snap_rx) = mpsc::channel::<TranscriptSnapshot>(64);
    let h = tokio::spawn(async move {
        run_session(backend, cfg, pcm_rx, snap_tx, SessionMetrics::default()).await
    });

    let mut t = 0.0;
    for chunk in audio.chunks(16000) {
        t += chunk.len() as f64 / 16000.0;
        pcm_tx
            .send(AudioChunk { samples: chunk.to_vec(), t_end: t })
            .await
            .unwrap();
    }
    drop(pcm_tx);

    let mut last = TranscriptSnapshot::default();
    while let Some(s) = snap_rx.recv().await {
        eprintln!("[snap] {:?} | buf={:?}", s.committed_text, s.buffer);
        last = s;
    }
    h.await.unwrap().unwrap();
    eprintln!("=== FINAL (Rust 네이티브): {}", last.committed_text);
    assert!(
        last.committed_text.to_lowercase().contains("country"),
        "전사에 country 없음: {:?}",
        last.committed_text
    );
}
