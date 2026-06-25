//! Q5_0 turbo 단독 벤치 — 풀패스 정확도(전사 저장→CER) + 스트리밍 레이턴시.
//! alt 가 쓰는 ggml-large-v3-turbo-q5_0 를 우리 파이프라인으로 측정.
//!
//! 실행: `cargo test -p app --test q5_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_whisper::WhisperRsBackend;
use asr_core::asr::{AsrConfig, StreamingAsrBackend};

const SR: usize = 16_000;
const OUT_DIR: &str = "/private/tmp/claude-501/-Users-kwonjunho-Desktop-work-tauri-stt-test/68e638b0-f4c7-4dbb-ba6f-f5c9ca7534ca/scratchpad/sweep";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Q5_0 모델 + 회의 음성 필요. --ignored 로 실행"]
async fn q5_bench() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = base.join("models/ggml/ggml-large-v3-turbo-q5_0.bin");
    assert!(model.exists(), "Q5_0 모델 없음: {model:?}");
    let mut r = hound::WavReader::open(base.join("test-data/meeting.wav")).expect("wav");
    let audio: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let audio_sec = audio.len() as f64 / SR as f64;
    std::fs::create_dir_all(OUT_DIR).ok();

    // 1) 풀패스 정확도 + RTF
    let be = WhisperRsBackend::load(&model, Some("ko".into())).expect("load");
    let _ = be.transcribe(&audio[..SR.min(audio.len())], "");
    let t0 = Instant::now();
    let toks = be.transcribe(&audio, "").expect("transcribe");
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let text: String = toks.iter().map(|t| t.text.as_str()).collect();
    std::fs::write(format!("{OUT_DIR}/ggml-large-v3-turbo-q5_0.txt"), &text).ok();
    eprintln!(
        "[Q5_0 풀패스] {:.1}s (RTF {:.3})",
        full_ms / 1000.0,
        full_ms / 1000.0 / audio_sec
    );

    // 2) 스트리밍 레이턴시(90s)
    let mut sb = asr_whisper::self_streaming_backend(base.join("models/ggml"));
    sb.configure(&AsrConfig {
        model_id: "ggml-large-v3-turbo-q5_0".into(),
        language: Some("ko".into()),
        ..AsrConfig::default()
    })
    .await
    .expect("configure");
    let n = 90usize.min(audio.len() / SR);
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
