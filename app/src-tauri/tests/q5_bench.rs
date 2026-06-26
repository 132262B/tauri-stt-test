//! Q5_0 turbo 단독 벤치 — 풀패스 정확도(전사 저장→CER) + 스트리밍 레이턴시.
//! alt 가 쓰는 ggml-large-v3-turbo-q5_0 를 우리 파이프라인으로 측정.
//!
//! 실행: `cargo test -p app --test q5_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_core::asr::{AsrConfig, AsrProfile};
use asr_whisper::{WhisperDecodeOptions, WhisperRsBackend};

const SR: usize = 16_000;
const OUT_DIR: &str = "/private/tmp/claude-501/-Users-kwonjunho-Desktop-work-tauri-stt-test/68e638b0-f4c7-4dbb-ba6f-f5c9ca7534ca/scratchpad/sweep";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Q5_0 모델 + 회의 음성 필요. --ignored 로 실행"]
async fn q5_bench() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = base.join("models/ggml/ggml-large-v3-turbo-q5_0.bin");
    let coreml = base.join("models/ggml/ggml-large-v3-turbo-encoder.mlmodelc");
    assert!(model.exists(), "Q5_0 모델 없음: {model:?}");
    assert!(
        coreml.exists() || allow_q5_without_coreml(),
        "Q5 realtime 벤치는 CoreML encoder가 필요함: {coreml:?}"
    );
    let mut r = hound::WavReader::open(base.join("test-data/meeting.wav")).expect("wav");
    let audio: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let bench_sec = std::env::var("BENCH_SEC")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(90)
        .max(1);
    let sample_limit = (bench_sec * SR).min(audio.len());
    let audio = &audio[..sample_limit];
    let audio_sec = audio.len() as f64 / SR as f64;
    std::fs::create_dir_all(OUT_DIR).ok();

    // 1) 풀패스 정확도 + RTF
    let opts = WhisperDecodeOptions::for_profile(AsrProfile::RealtimeQ5);
    let be = WhisperRsBackend::load_with_options(&model, Some("ko".into()), opts).expect("load");
    let _ = be.transcribe_text(&audio[..SR.min(audio.len())], "", opts);
    let t0 = Instant::now();
    let text = be.transcribe_text(&audio, "", opts).expect("transcribe");
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(format!("{OUT_DIR}/ggml-large-v3-turbo-q5_0.txt"), &text).ok();
    eprintln!(
        "[Q5_0 풀패스] {:.1}s (RTF {:.3})",
        full_ms / 1000.0,
        full_ms / 1000.0 / audio_sec
    );

    // 2) 스트리밍 레이턴시(BENCH_SEC)
    let mut sb = asr_whisper::self_streaming_backend(base.join("models/ggml"));
    sb.configure(&AsrConfig {
        model_id: "ggml-large-v3-turbo-q5_0".into(),
        language: Some("ko".into()),
        profile: AsrProfile::RealtimeQ5,
        ..AsrConfig::default()
    })
    .await
    .expect("configure");
    sb.warmup().await.expect("warmup");
    let n = bench_sec.min(audio.len() / SR);
    let mut ticks: Vec<f64> = Vec::new();
    let (mut lag, mut lagn, mut over) = (0.0f64, 0usize, 0usize);
    for i in 0..n {
        sb.insert_audio_chunk(&audio[i * SR..(i + 1) * SR], (i + 1) as f64);
        let t = Instant::now();
        let toks = sb.process_iter(false).await.expect("iter");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        ticks.push(ms);
        if ms > 1000.0 {
            over += 1;
        }
        for tk in &toks {
            lag += (i + 1) as f64 - tk.end;
            lagn += 1;
        }
    }
    ticks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = ticks.iter().sum::<f64>() / ticks.len() as f64;
    let p95 = ticks[((ticks.len() as f64 * 0.95) as usize).min(ticks.len() - 1)];
    eprintln!(
        "[Q5_0 스트리밍 {n}s] 평균틱 {:.0}ms, p95 {:.0}ms, 최대 {:.0}ms, 확정지연 {:.1}s, 미추종 {}/{}",
        avg,
        p95,
        ticks.last().unwrap(),
        if lagn > 0 { lag / lagn as f64 } else { 0.0 },
        over,
        n
    );
}

fn allow_q5_without_coreml() -> bool {
    std::env::var("ASR_Q5_ALLOW_NO_COREML")
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}
