//! 화자분리 병목 가시화 — 오디오 길이별 diar 소요시간/RTF.
//! 누적 오디오를 매 패스 통째로 재분석하므로 길수록 느려진다(구조적 비용).
//! 실행: `DIAR_DEBUG=1 cargo test -p diar-pyannote --test diar_scaling -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_core::diar::Diarizer;
use diar_pyannote::PyannoteDiarizer;

#[test]
#[ignore = "모델+음성 필요"]
fn diar_time_vs_length() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut r = hound::WavReader::open(base.join("test-data/meeting.wav")).expect("wav");
    let all: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let mut d = PyannoteDiarizer::with_paths(base.join("models"), None).expect("diar");

    eprintln!("\n##### 화자분리 소요시간 vs 오디오 길이 #####");
    for secs in [30usize, 60, 120, 300, 599] {
        let n = (secs * 16_000).min(all.len());
        let audio = &all[..n];
        let t0 = Instant::now();
        let segs = d.diarize(audio);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  {secs:>3}초 오디오 → {:>5.0}ms  (RTF {:.2}, {}세그먼트)",
            ms,
            ms / 1000.0 / secs as f64,
            segs.len()
        );
    }
    eprintln!(
        "\n참고: 앱은 ~10초마다 '누적 오디오 전체'를 재분석 → 길어질수록 위 시간만큼 매번 듦."
    );
}
