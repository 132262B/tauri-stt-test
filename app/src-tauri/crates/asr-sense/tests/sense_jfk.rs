//! 키스톤: SenseVoice(다국어, Rust)로 JFK 전사. Python 0.
//! 실행: `cargo test -p asr-sense --test sense_jfk -- --ignored --nocapture`

use std::path::PathBuf;

use asr_sense::SenseVoiceBackend;
use asr_core::asr::SelfStreamingBackend;

#[test]
#[ignore = "SenseVoice 모델 필요. --ignored 로 실행"]
fn sense_voice_transcribes_jfk() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = base.join("models/sense");
    assert!(model_dir.join("model.onnx").exists(), "SenseVoice 모델 없음: {model_dir:?}");

    let mut r = hound::WavReader::open(base.join("test-data/jfk.wav")).expect("wav");
    let audio: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();

    let mut b = SenseVoiceBackend::new(&model_dir, Some("en".into())).expect("sense");
    let text = b.transcribe_full(&audio, "").expect("transcribe");
    eprintln!("=== SenseVoice: {text}");
    assert!(text.to_lowercase().contains("country"), "전사에 'country' 없음: {text:?}");
}
