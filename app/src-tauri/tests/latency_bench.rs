//! 스트리밍 레이턴시 벤치 — 모델별 1초 청크 투입 시 틱 지연·확정 지연 측정.
//! 평균/95퍼센타일/최대 틱(ms) + 평균 확정 지연(말한 뒤 굳기까지, 초) + 실시간 미추종.
//!
//! 실행: `cargo test -p app --test latency_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_core::asr::{AsrConfig, StreamingAsrBackend};

const SR: usize = 16_000;

struct Lat {
    name: &'static str,
    avg_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    commit_lag: f64,
    over_rt: usize,
    ticks: usize,
}

async fn measure(
    name: &'static str,
    mut backend: Box<dyn StreamingAsrBackend>,
    cfg: &AsrConfig,
    audio: &[f32],
    sec: usize,
) -> Option<Lat> {
    if backend.configure(cfg).await.is_err() {
        eprintln!("[skip] {name} configure 실패");
        return None;
    }
    let n = sec.min(audio.len() / SR);
    let mut ticks_ms: Vec<f64> = Vec::new();
    let mut lag_sum = 0.0;
    let mut lag_n = 0usize;
    let mut over = 0usize;
    for i in 0..n {
        backend.insert_audio_chunk(&audio[i * SR..(i + 1) * SR], (i + 1) as f64);
        let t0 = Instant::now();
        let toks = backend.process_iter(false).await.ok()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        ticks_ms.push(ms);
        if ms > 1000.0 {
            over += 1;
        }
        for tk in &toks {
            lag_sum += (i + 1) as f64 - tk.end;
            lag_n += 1;
        }
    }
    ticks_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg = ticks_ms.iter().sum::<f64>() / ticks_ms.len().max(1) as f64;
    let p95 = ticks_ms[((ticks_ms.len() as f64 * 0.95) as usize).min(ticks_ms.len() - 1)];
    let max = *ticks_ms.last().unwrap_or(&0.0);
    eprintln!("[done] {name} (스트리밍 {n}s)");
    Some(Lat {
        name,
        avg_ms: avg,
        p95_ms: p95,
        max_ms: max,
        commit_lag: if lag_n > 0 {
            lag_sum / lag_n as f64
        } else {
            0.0
        },
        over_rt: over,
        ticks: n,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "회의 음성 + 모델 필요. --ignored 로 실행"]
async fn latency_bench() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut r = hound::WavReader::open(base.join("test-data/meeting.wav")).expect("wav");
    let audio: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let ggml = base.join("models/ggml");
    let cfg = |m: &str| AsrConfig {
        model_id: m.into(),
        language: Some("ko".into()),
        ..AsrConfig::default()
    };

    let mut out: Vec<Lat> = Vec::new();

    // Metal whisper 계열 + SenseVoice 는 120s, CPU Qwen 은 60s(시간 절약, 정상상태 수렴엔 충분).
    for (name, model) in [
        ("ggml-tiny", "ggml-tiny"),
        ("ggml-base", "ggml-base"),
        ("ggml-small", "ggml-small"),
        ("ggml-large-v3-turbo", "ggml-large-v3-turbo"),
        ("ggml-large-v3", "ggml-large-v3"),
    ] {
        let be = asr_whisper::self_streaming_backend(&ggml);
        if let Some(l) = measure(name, be, &cfg(model), &audio, 90).await {
            out.push(l);
        }
    }
    if base.join("models/sense/model.onnx").exists() {
        if let Ok(be) = asr_sense::streaming_backend(base.join("models/sense"), Some("ko".into())) {
            if let Some(l) = measure("sensevoice", be, &cfg("sensevoice"), &audio, 90).await {
                out.push(l);
            }
        }
    }
    // Qwen 은 CPU 라 느림 → 정상상태 수렴에 충분한 짧은 구간으로.
    for (name, dir, spec, sec) in [
        ("qwen-0.6b", "models/qwen", &asr_qwen::QWEN_06B, 45usize),
    ] {
        let d = base.join(dir);
        if !d.join("config.json").exists() {
            continue;
        }
        if let Ok(be) = asr_qwen::streaming_backend(&d, spec, Some("ko".into())) {
            if let Some(l) = measure(name, be, &cfg(name), &audio, sec).await {
                out.push(l);
            }
        }
    }

    eprintln!("\n============ 스트리밍 레이턴시 ============");
    eprintln!(
        "{:<22} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "model", "평균틱", "p95틱", "최대틱", "확정지연", "미추종"
    );
    for l in &out {
        eprintln!(
            "{:<22} {:>7.0}ms {:>7.0}ms {:>7.0}ms {:>8.1}s {:>6}/{}",
            l.name, l.avg_ms, l.p95_ms, l.max_ms, l.commit_lag, l.over_rt, l.ticks
        );
    }
    eprintln!("==========================================\n");
}
