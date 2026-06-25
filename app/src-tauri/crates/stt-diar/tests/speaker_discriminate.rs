//! 키스톤: Rust 화자 분리(sherpa-onnx)가 2화자를 구분하고 재식별하는지.
//! 실행: `cargo test -p stt-diar --test speaker_discriminate -- --ignored --nocapture`

use std::path::PathBuf;

use stt_diar::OnlineDiarizer;

fn read_wav(p: &PathBuf) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).expect("wav");
    r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
}

#[test]
#[ignore = "화자 모델 필요. --ignored 로 실행"]
fn two_speakers_distinguished_and_reidentified() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = base.join("models/speaker/campplus.onnx");
    assert!(model.exists(), "화자 모델 없음: {model:?}");
    let a = read_wav(&base.join("test-data/spk_a.wav")); // Yuna(ko)
    let b = read_wav(&base.join("test-data/spk_b.wav")); // Daniel(en)

    let mut diar = OnlineDiarizer::new(&model, 0.5).expect("diarizer");
    let s_a1 = diar.assign(&a).expect("a1");
    let s_b = diar.assign(&b).expect("b");
    let s_a2 = diar.assign(&a).expect("a2");
    eprintln!("Yuna#1={s_a1}  Daniel={s_b}  Yuna#2={s_a2}");

    assert_ne!(s_a1, s_b, "서로 다른 화자가 같은 id로 묶임");
    assert_eq!(s_a1, s_a2, "같은 화자(Yuna)가 재식별되지 않음");
}
