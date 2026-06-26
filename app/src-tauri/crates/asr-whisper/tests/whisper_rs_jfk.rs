//! Rust 네이티브 Whisper 전사 검증(whisper.cpp/Metal). Python 0줄.
//! 실행: `cargo test -p asr-whisper --test whisper_rs_jfk -- --ignored --nocapture`

use std::path::PathBuf;

use asr_whisper::WhisperRsBackend;

#[test]
#[ignore = "ggml 모델 필요. --ignored 로 실행"]
fn jfk_rust_native() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = base.join("models/ggml/ggml-base.bin");
    let wav = base.join("test-data/jfk.wav");
    assert!(model.exists(), "ggml 모델 없음: {model:?}");
    assert!(wav.exists(), "오디오 없음: {wav:?}");

    let mut reader = hound::WavReader::open(&wav).expect("wav open");
    assert_eq!(reader.spec().sample_rate, 16_000);
    let audio: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let backend = WhisperRsBackend::load(&model, Some("en".into())).expect("load");
    let tokens = backend.transcribe(&audio, "").expect("transcribe");
    let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
    eprintln!("=== RUST-NATIVE WHISPER: {text}");
    let low = text.to_lowercase();
    assert!(low.contains("country"), "전사에 'country' 없음: {text:?}");
}

/// 자동 언어감지(language=None)가 한국어를 영어가 아닌 한글로 전사하는지 검증.
/// (회귀 방지: whisper.cpp 기본 언어 "en" 으로 한글이 영어로 새던 버그)
#[test]
#[ignore = "ggml 모델 필요. --ignored 로 실행"]
fn korean_autodetect_not_english() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = base.join("models/ggml/ggml-base.bin");
    let wav = base.join("test-data/ko_test.wav");
    assert!(wav.exists(), "한국어 오디오 없음: {wav:?}");
    let mut reader = hound::WavReader::open(&wav).expect("wav open");
    let audio: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    // language=None → "auto" 감지 경로.
    let backend = WhisperRsBackend::load(&model, None).expect("load");
    let _ = backend.transcribe(&audio, ""); // 워밍업
    let audio_sec = audio.len() as f64 / 16_000.0;
    let t0 = std::time::Instant::now();
    let tokens = backend.transcribe(&audio, "").expect("transcribe");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "=== WHISPER base ko ({audio_sec:.1}s audio): {ms:.0}ms  RTF={:.2}",
        ms / 1000.0 / audio_sec
    );
    let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
    eprintln!("=== KO autodetect: {text}");
    let hangul = text
        .chars()
        .filter(|c| ('\u{AC00}'..='\u{D7A3}').contains(c))
        .count();
    assert!(
        hangul >= 5,
        "한글이 거의 없음(영어로 샘) → 자동감지 실패: {text:?}"
    );
}

/// turbo·large-v3 가 whisper-rs 0.16 에서 로드/전사되는지 검증.
#[test]
#[ignore = "큰 모델 필요. --ignored 로 실행"]
fn turbo_and_large_load() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut r = hound::WavReader::open(base.join("test-data/jfk.wav")).expect("wav");
    let audio: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    for name in ["ggml-large-v3-turbo.bin", "ggml-large-v3.bin"] {
        let model = base.join("models/ggml").join(name);
        if !model.exists() {
            eprintln!("[skip] {name} 없음");
            continue;
        }
        let backend = WhisperRsBackend::load(&model, Some("en".into()))
            .unwrap_or_else(|e| panic!("{name} 로드 실패: {e}"));
        let toks = backend.transcribe(&audio, "").expect("transcribe");
        let text: String = toks.iter().map(|t| t.text.as_str()).collect();
        eprintln!("=== {name}: {text}");
        assert!(
            text.to_lowercase().contains("country"),
            "{name} 전사 실패: {text:?}"
        );
    }
}
