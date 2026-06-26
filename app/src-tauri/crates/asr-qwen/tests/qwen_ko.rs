//! Qwen3-ASR-0.6B(antirez/qwen-asr, C+FFI) 한국어 전사 검증.
//! 실행: `cargo test -p asr-qwen --test qwen_ko -- --ignored --nocapture`

use std::path::PathBuf;

use asr_core::asr::SelfStreamingBackend;
use asr_qwen::{QwenBackend, QWEN_06B};

#[test]
#[ignore = "Qwen 모델(1.7G) 필요. --ignored 로 실행"]
fn qwen_transcribes_korean() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = base.join("models/qwen");
    let wav = base.join("test-data/ko_test.wav");
    assert!(
        model_dir.join("model.safetensors").exists(),
        "Qwen 모델 없음"
    );
    assert!(wav.exists(), "한국어 오디오 없음: {wav:?}");

    let mut reader = hound::WavReader::open(&wav).expect("wav open");
    assert_eq!(reader.spec().sample_rate, 16_000);
    let audio: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();

    let mut backend = QwenBackend::new(&model_dir, &QWEN_06B, Some("ko".into())).expect("load");
    // 워밍업 1회 후 측정(순수 추론 시간 = 로드 제외).
    let _ = backend.transcribe_full(&audio, "");
    let audio_sec = audio.len() as f64 / 16_000.0;
    let t0 = std::time::Instant::now();
    let text = backend.transcribe_full(&audio, "").expect("transcribe");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "=== QWEN3-ASR ko ({audio_sec:.1}s audio): {ms:.0}ms  RTF={:.2}",
        ms / 1000.0 / audio_sec
    );
    eprintln!("=== QWEN3-ASR ko: {text}");
    let hangul = text
        .chars()
        .filter(|c| ('\u{AC00}'..='\u{D7A3}').contains(c))
        .count();
    assert!(hangul >= 5, "한글이 거의 없음 → 전사 실패: {text:?}");
}
