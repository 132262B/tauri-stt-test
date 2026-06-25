//! mlx-rs API 검증: 가중치 로딩 + conv1d 레이아웃.
//! 실행: `cargo test -p asr-mlx --test api_check -- --ignored --nocapture`

use std::path::PathBuf;

#[test]
#[ignore = "whisper-base 모델 필요"]
fn weights_and_conv() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = base.join("models/mlx/base/model.safetensors");
    assert!(model.exists(), "모델 없음: {model:?}");

    let w = asr_mlx::load_weights(&model).expect("load");
    eprintln!("가중치 키 {}개", w.len());
    // 인코더/디코더 핵심 키 몇 개 + shape 출력
    for k in [
        "model.encoder.conv1.weight",
        "model.encoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.encoder_attn.k_proj.weight",
        "model.decoder.embed_tokens.weight",
    ] {
        match w.get(k) {
            Some(a) => eprintln!("  {k}: {:?}", a.shape()),
            None => eprintln!("  {k}: (없음)"),
        }
    }
    let out_shape = asr_mlx::conv1d_check(&w).expect("conv1d");
    eprintln!("conv1d 출력 shape: {out_shape:?} (기대 [1, 100, 512])");
    assert_eq!(out_shape, vec![1, 100, 512], "conv1d 레이아웃 불일치");
}
