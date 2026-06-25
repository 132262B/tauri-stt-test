//! 실제 회의 음성으로 실시간 스트리밍 시뮬레이션 + 측정.
//! 1초 청크를 순차 투입하며 매 틱 process_iter 지연을 측정한다.
//! 실행: `cargo test -p asr-whisper --test meeting_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_whisper::WhisperSelfBackend;
use asr_core::asr::{AsrConfig, SelfStreamingProcessor, StreamingAsrBackend};

#[tokio::test]
#[ignore = "meeting.wav + 모델 필요. --ignored 로 실행"]
async fn meeting_streaming_bench() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let wav = base.join("test-data/meeting.wav");
    assert!(wav.exists(), "meeting.wav 없음: {wav:?}");
    let mut reader = hound::WavReader::open(&wav).expect("wav");
    let audio: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // 측정 구간(초). 환경변수 BENCH_SEC 로 조절(기본 120s).
    let n_sec: usize = std::env::var("BENCH_SEC").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
    let n_sec = n_sec.min(audio.len() / 16_000);

    let model_id = std::env::var("BENCH_MODEL").unwrap_or_else(|_| "ggml-base".into());
    eprintln!("model: {model_id}");
    let mut proc = SelfStreamingProcessor::new(WhisperSelfBackend::new(base.join("models/ggml")));
    proc.configure(&AsrConfig {
        model_id: model_id.clone(),
        language: Some("ko".into()),
        ..AsrConfig::default()
    })
    .await
    .expect("configure");

    let mut committed = String::new();
    let mut total_ms = 0.0f64;
    let mut max_ms = 0.0f64;
    let mut over_rt = 0usize; // 1초 틱을 초과한(실시간 추종 실패) 횟수
    let mut commit_lag_sum = 0.0f64;
    let mut commit_events = 0usize;

    // 틱 간격(초): 한 번 전사할 때마다 투입할 오디오 길이. 큰 모델은 2~3초로 빈도를 낮춰
    // 전사 1회 비용을 분산 → 실시간 추종. 기본 1초.
    let tick: usize = std::env::var("TICK_SEC").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let budget_ms = tick as f64 * 1000.0;
    let steps = n_sec / tick;
    eprintln!("tick={tick}s, budget={budget_ms:.0}ms/tick");
    for i in 0..steps {
        let seg = &audio[i * tick * 16_000..(i + 1) * tick * 16_000];
        proc.insert_audio_chunk(seg, ((i + 1) * tick) as f64);
        let t0 = Instant::now();
        let toks = proc.process_iter(false).await.expect("iter");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;
        if ms > max_ms {
            max_ms = ms;
        }
        if ms > budget_ms {
            over_rt += 1;
        }
        for tk in &toks {
            committed.push_str(&tk.text);
            commit_lag_sum += ((i + 1) * tick) as f64 - tk.end;
            commit_events += 1;
        }
    }
    let n_sec = steps * tick;

    let audio_ms = n_sec as f64 * 1000.0;
    eprintln!("\n===== 회의 스트리밍 벤치 ({n_sec}s 구간) =====");
    eprintln!("총 추론시간 {:.0}ms / 오디오 {:.0}ms → 전체 RTF {:.2}", total_ms, audio_ms, total_ms / audio_ms);
    eprintln!("틱당 최대 {:.0}ms, 평균 {:.0}ms", max_ms, total_ms / steps as f64);
    eprintln!("실시간 미추종 틱(>{budget_ms:.0}ms): {over_rt}/{steps}");
    if commit_events > 0 {
        eprintln!("평균 확정 지연 ≈ {:.1}s", commit_lag_sum / commit_events as f64);
    }
    eprintln!("--- 확정 전사(앞 600자) ---\n{}", committed.chars().take(600).collect::<String>());
    eprintln!("===== 끝 =====\n");
}
