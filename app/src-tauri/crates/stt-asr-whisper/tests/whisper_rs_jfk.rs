//! Rust 네이티브 Whisper 전사 검증(whisper.cpp/Metal). Python 0줄.
//! 실행: `cargo test -p stt-asr-whisper --test whisper_rs_jfk -- --ignored --nocapture`

use std::path::PathBuf;

use stt_asr_whisper::WhisperRsBackend;

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
