//! MLX 인코더 속도 측정 — candle(turbo ~1.8s)/whisper.cpp 대비 GO/NO-GO 게이트.
//! 30초 더미 mel 로 인코더 forward 1회 시간(강제 eval 포함).
//! 실행: `cargo test -p asr-mlx --test encode_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::Array;

fn cfg_val(dir: &PathBuf, key: &str) -> i64 {
    let s = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    v[key].as_i64().unwrap()
}

fn bench(name: &str, dir: PathBuf) {
    if !dir.join("model.safetensors").exists() {
        eprintln!("[skip] {name} 모델 없음");
        return;
    }
    let n_mels = cfg_val(&dir, "num_mel_bins") as i32;
    let n_layers = cfg_val(&dir, "encoder_layers") as i32;
    let n_head = cfg_val(&dir, "encoder_attention_heads") as i32;

    let w = asr_mlx::load_weights(dir.join("model.safetensors")).expect("load");
    let mel = Array::zeros::<f32>(&[1, n_mels, 3000]).expect("mel"); // 30초

    // 워밍업(커널 컴파일/캐시 포함)
    let out = asr_mlx::encode(&w, &mel, n_layers, n_head).expect("encode");
    out.eval().expect("eval");

    let t0 = Instant::now();
    let out = asr_mlx::encode(&w, &mel, n_layers, n_head).expect("encode");
    out.eval().expect("eval");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "=== {name}: 인코더 30초 forward = {ms:.0}ms, 출력 {:?} ({n_layers}층/{n_head}헤드)",
        out.shape()
    );
}

#[test]
#[ignore = "모델 필요"]
fn mlx_encoder_speed() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    eprintln!("\n##### MLX 인코더 속도(30초 mel) #####");
    bench("base", base.join("models/mlx/base"));
    bench("turbo", base.join("models/mlx/turbo"));
    eprintln!("(비교: candle turbo 인코더 ~1800ms, whisper.cpp turbo ~500-900ms)");
}
