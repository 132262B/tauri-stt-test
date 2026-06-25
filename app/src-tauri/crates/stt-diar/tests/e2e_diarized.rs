//! 전체 Rust 경로 + 화자분리 end-to-end: 2화자 오디오 → 전사 라인에 2개 이상 화자 라벨.
//! Python 0. 실행: `cargo test -p stt-diar --test e2e_diarized -- --ignored --nocapture`

use std::collections::HashSet;
use std::path::PathBuf;

use stt_asr_whisper::WhisperStreamingBackend;
use stt_core::asr::{AsrConfig, StreamingAsrBackend};
use stt_core::diar::Diarizer;
use stt_core::metrics::SessionMetrics;
use stt_core::output::TranscriptSnapshot;
use stt_core::pipeline::{run_session, AudioChunk};
use stt_diar::OnlineDiarizer;
use tokio::sync::mpsc;

fn read_wav(p: &PathBuf) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).expect("wav");
    r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ggml+화자 모델 필요. --ignored 로 실행"]
async fn two_speakers_get_distinct_labels() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let a = read_wav(&base.join("test-data/spk_a.wav")); // Yuna(ko)
    let b = read_wav(&base.join("test-data/spk_b.wav")); // Daniel(en)
    // Yuna → Daniel → Yuna (재식별 포함)
    let mut audio = a.clone();
    audio.extend(std::iter::repeat(0.0).take(8000));
    audio.extend_from_slice(&b);
    audio.extend(std::iter::repeat(0.0).take(8000));
    audio.extend_from_slice(&a);

    let backend: Box<dyn StreamingAsrBackend> =
        Box::new(WhisperStreamingBackend::new(base.join("models/ggml")));
    let diar: Option<Box<dyn Diarizer>> = Some(Box::new(
        OnlineDiarizer::new(base.join("models/speaker/campplus.onnx"), 0.5).expect("diar"),
    ));
    let cfg = AsrConfig {
        model_id: "ggml-base".into(),
        language: None,
        trimming_sec: 15.0,
    };
    let (pcm_tx, pcm_rx) = mpsc::channel::<AudioChunk>(64);
    let (snap_tx, mut snap_rx) = mpsc::channel::<TranscriptSnapshot>(64);
    let h = tokio::spawn(async move {
        run_session(backend, cfg, pcm_rx, snap_tx, SessionMetrics::default(), diar).await
    });

    let mut t = 0.0;
    for chunk in audio.chunks(16000) {
        t += chunk.len() as f64 / 16000.0;
        pcm_tx.send(AudioChunk { samples: chunk.to_vec(), t_end: t }).await.unwrap();
    }
    drop(pcm_tx);

    let mut last = TranscriptSnapshot::default();
    while let Some(s) = snap_rx.recv().await {
        last = s;
    }
    h.await.unwrap().unwrap();

    let speakers: HashSet<u32> = last.lines.iter().filter_map(|l| l.speaker).collect();
    for l in &last.lines {
        eprintln!("[{:?}] {}", l.speaker, l.text);
    }
    eprintln!("=== 화자 수: {} ===", speakers.len());
    assert!(speakers.len() >= 2, "최소 2명 화자 라벨이 나와야 함: {speakers:?}");
}
