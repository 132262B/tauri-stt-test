//! Q5 decode option probe.
//!
//! 실행:
//! `cargo test -p asr-whisper --test q5_decode_probe -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_core::asr::{AsrConfig, AsrProfile};
use asr_whisper::{WhisperDecodeOptions, WhisperRsBackend};

#[test]
#[ignore = "Q5_0 모델 + CoreML encoder 필요"]
fn q5_decode_probe() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = q5_model_path(&base);
    assert!(model.exists(), "Q5 model missing: {model:?}");

    let files = ["ko_test.wav", "jfk.wav", "meeting.wav"];
    let profiles = [
        (
            "realtime_old",
            WhisperDecodeOptions::for_profile(AsrProfile::RealtimeQ5),
        ),
        ("windowed", WhisperDecodeOptions::for_windowed_q5()),
        ("balanced", WhisperDecodeOptions::default()),
        (
            "balanced_cpu",
            WhisperDecodeOptions {
                use_gpu: false,
                ..WhisperDecodeOptions::default()
            },
        ),
    ];

    for (profile_name, opts) in profiles {
        eprintln!("\n=== profile={profile_name} opts={opts:?} ===");
        let be = WhisperRsBackend::load_with_options(&model, None, opts).expect("load");
        for file in files {
            let path = base.join("test-data").join(file);
            let audio = read_wav(&path);
            let samples = if file == "meeting.wav" {
                &audio[..audio.len().min(30 * 16_000)]
            } else {
                &audio[..]
            };
            let t0 = Instant::now();
            let text = be.transcribe_text(samples, "", opts).expect("transcribe");
            eprintln!(
                "{file}: {:.1}s audio, {:.0}ms, chars={}, text={:?}",
                samples.len() as f64 / 16_000.0,
                t0.elapsed().as_secs_f64() * 1000.0,
                text.chars().count(),
                text.chars().take(240).collect::<String>()
            );
        }
    }
}

#[test]
#[ignore = "Q5_0 모델 + CoreML encoder 필요"]
fn q5_prefix_probe() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = q5_model_path(&base);
    let audio = read_wav(&probe_audio_path(&base));
    let opts = WhisperDecodeOptions::for_windowed_q5();
    let be = WhisperRsBackend::load_with_options(&model, None, opts).expect("load");

    for sec in [8usize, 12, 16, 20, 24, 30, 40] {
        let samples = &audio[..audio.len().min(sec * 16_000)];
        let t0 = Instant::now();
        let text = be.transcribe_text(samples, "", opts).expect("transcribe");
        eprintln!(
            "{sec:>2}s: {:>6.0}ms chars={:<4} {:?}",
            t0.elapsed().as_secs_f64() * 1000.0,
            text.chars().count(),
            text.chars().take(180).collect::<String>()
        );
    }
}

#[test]
#[ignore = "Q5_0 모델 + CoreML encoder 필요"]
fn q5_slice_probe() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = q5_model_path(&base);
    let audio = read_wav(&probe_audio_path(&base));
    let opts = WhisperDecodeOptions::for_windowed_q5();
    let be = WhisperRsBackend::load_with_options(&model, None, opts).expect("load");

    for start_sec in [0usize, 5, 10, 15, 20, 25] {
        let start = start_sec * 16_000;
        let end = (start + 8 * 16_000).min(audio.len());
        let t0 = Instant::now();
        let text = be
            .transcribe_text(&audio[start..end], "", opts)
            .expect("transcribe");
        eprintln!(
            "{start_sec:>2}-{:<2}s: {:>6.0}ms chars={:<4} {:?}",
            start_sec + 8,
            t0.elapsed().as_secs_f64() * 1000.0,
            text.chars().count(),
            text.chars().take(180).collect::<String>()
        );
    }
}

#[test]
#[ignore = "Q5_0 모델 필요"]
fn q5_latency_probe() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = q5_model_path(&base);
    let audio = read_wav(&probe_audio_path(&base));
    let opts = WhisperDecodeOptions::for_windowed_q5();
    let be = WhisperRsBackend::load_with_options(&model, None, opts).expect("load");

    for len_sec in [5usize, 6, 7] {
        eprintln!("\n=== len={len_sec}s ===");
        for start_sec in [0usize, 3, 6, 9, 12, 15, 20, 25] {
            let start = start_sec * 16_000;
            let end = (start + len_sec * 16_000).min(audio.len());
            let t0 = Instant::now();
            let text = be
                .transcribe_text(&audio[start..end], "", opts)
                .expect("transcribe");
            eprintln!(
                "{start_sec:>2}-{:<2}s: {:>6.0}ms chars={:<4} {:?}",
                start_sec + len_sec,
                t0.elapsed().as_secs_f64() * 1000.0,
                text.chars().count(),
                text.chars().take(140).collect::<String>()
            );
        }
    }
}

fn q5_model_path(base: &std::path::Path) -> PathBuf {
    std::env::var("Q5_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| base.join("models/ggml/ggml-large-v3-turbo-q5_0.bin"))
}

fn probe_audio_path(base: &std::path::Path) -> PathBuf {
    std::env::var("AUDIO_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| base.join("test-data/meeting.wav"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Q5_0 모델 + CoreML encoder 필요"]
async fn q5_streaming_backlog_smoke() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let audio = read_wav(&probe_audio_path(&base));
    let mut backend = asr_whisper::self_streaming_backend(base.join("models/ggml"));
    backend
        .configure(&AsrConfig {
            model_id: "ggml-large-v3-turbo-q5_0".into(),
            language: Some("ko".into()),
            profile: AsrProfile::RealtimeQ5,
            ..AsrConfig::default()
        })
        .await
        .expect("configure");
    backend.warmup().await.expect("warmup");

    let mut committed = String::new();
    for i in 0..8 {
        backend.insert_audio_chunk(&audio[i * 16_000..(i + 1) * 16_000], (i + 1) as f64);
    }
    let t0 = Instant::now();
    let toks = backend.process_iter(false).await.expect("iter 0-8");
    eprintln!(
        "iter 0-8 took={:.0}ms text={:?}",
        t0.elapsed().as_secs_f64() * 1000.0,
        toks.iter().map(|t| t.text.as_str()).collect::<String>()
    );
    for tk in toks {
        committed.push_str(&tk.text);
    }

    for i in 8..20 {
        backend.insert_audio_chunk(&audio[i * 16_000..(i + 1) * 16_000], (i + 1) as f64);
    }

    let t1 = Instant::now();
    let toks = backend.process_iter(false).await.expect("iter 5-13");
    let text_4_12 = toks.iter().map(|t| t.text.as_str()).collect::<String>();
    eprintln!(
        "iter backlog next took={:.0}ms text={text_4_12:?}",
        t1.elapsed().as_secs_f64() * 1000.0
    );
    for tk in toks {
        committed.push_str(&tk.text);
    }

    let t2 = Instant::now();
    let toks = backend.process_iter(false).await.expect("iter 10-18");
    let text_8_16 = toks.iter().map(|t| t.text.as_str()).collect::<String>();
    eprintln!(
        "iter backlog next2 took={:.0}ms text={text_8_16:?}",
        t2.elapsed().as_secs_f64() * 1000.0
    );
    for tk in toks {
        committed.push_str(&tk.text);
    }

    eprintln!("committed={committed:?}");
    assert!(
        committed.contains("월요일"),
        "first window missing: {committed}"
    );
    assert!(
        committed.contains("오늘") || text_4_12.contains("오늘"),
        "5-13s content was skipped: {committed}"
    );
    assert!(
        committed.contains("스프린트") || text_8_16.contains("스프린트"),
        "10-18s content was skipped: {committed}"
    );
    assert!(
        !committed.contains("이는"),
        "Q5 language fallback produced hallucination: {committed}"
    );
}

fn read_wav(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 16_000);
    reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32768.0)
        .collect()
}
