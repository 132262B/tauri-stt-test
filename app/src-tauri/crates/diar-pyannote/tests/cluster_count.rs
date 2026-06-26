//! 화자 수가 2로 고정인지 확인 — 여러 num_clusters/threshold 로 검출 화자 수 비교.
//! 실행: `cargo test -p diar-pyannote --test cluster_count -- --ignored --nocapture`

use std::collections::BTreeSet;
use std::path::PathBuf;

use sherpa_rs::diarize::{Diarize, DiarizeConfig};

fn run(seg: &PathBuf, emb: &PathBuf, audio: &[f32], num: Option<i32>, thr: f32) -> (usize, usize) {
    let mut d = Diarize::new(
        seg,
        emb,
        DiarizeConfig {
            num_clusters: num,
            threshold: Some(thr),
            ..Default::default()
        },
    )
    .expect("diar");
    let segs = d.compute(audio.to_vec(), None).expect("compute");
    let n = segs
        .iter()
        .map(|s| s.speaker)
        .collect::<BTreeSet<_>>()
        .len();
    (n, segs.len())
}

#[test]
#[ignore = "모델+음성 필요"]
fn cluster_count_sweep() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let seg = base.join("models/diar/sherpa-onnx-pyannote-segmentation-3-0/model.onnx");
    let emb = base.join("models/speaker/campplus.onnx");
    let mut r = hound::WavReader::open(base.join("test-data/meeting.wav")).expect("wav");
    let all: Vec<f32> = r
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let audio = &all[..(150 * 16_000).min(all.len())]; // 앞 150초(4명 다 등장)

    eprintln!("\n=== 화자 수 강제/자동 비교 (앞 150s) ===");
    for num in [Some(2), Some(3), Some(4), Some(6)] {
        let (n, segs) = run(&seg, &emb, audio, num, 0.5);
        eprintln!("num_clusters={:?} → 검출 {}명, {}세그먼트", num, n, segs);
    }
    for thr in [0.3f32, 0.5, 0.7, 0.9] {
        let (n, segs) = run(&seg, &emb, audio, None, thr);
        eprintln!("자동 threshold={thr} → 검출 {}명, {}세그먼트", n, segs);
    }
}
