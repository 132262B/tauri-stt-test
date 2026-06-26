//! ASR: Whisper on MLX (mlx-rs). 인코더 forward 구현(속도 게이트 측정용).
//! 디코더/디코드루프/mel 은 인코더 속도 검증 후 단계적으로 추가.
//!
//! mlx-rs 관례: 산술 연산자(+ - * /)는 Array 를 반환(에러 시 panic),
//! matmul/conv1d/softmax/sqrt/reshape/transpose_axes/mean_axes/var_axes 등은 Result 를 반환.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_rs::Array;

pub type Weights = HashMap<String, Array>;

const EPS: f32 = 1e-5;

pub fn load_weights(path: impl AsRef<Path>) -> Result<Weights, String> {
    Array::load_safetensors(path.as_ref().to_str().unwrap()).map_err(|e| format!("load: {e}"))
}

fn get<'a>(w: &'a Weights, k: &str) -> Result<&'a Array, String> {
    w.get(k).ok_or_else(|| format!("가중치 없음: {k}"))
}

/// y = x @ W^T (+b). HF Linear 가중치는 [out, in].
fn linear(x: &Array, w: &Array, b: Option<&Array>) -> Result<Array, String> {
    let wt = w.transpose_axes(&[1, 0]).map_err(|e| e.to_string())?;
    let y = x.matmul(&wt).map_err(|e| e.to_string())?;
    Ok(match b {
        Some(b) => &y + b,
        None => y,
    })
}

/// LayerNorm(마지막 축).
fn layer_norm(x: &Array, w: &Array, b: &Array) -> Result<Array, String> {
    let mean = x.mean_axes(&[-1], true).map_err(|e| e.to_string())?;
    let var = x.var_axes(&[-1], true, 0).map_err(|e| e.to_string())?;
    let xc = x - &mean;
    let denom = ops::sqrt(&(&var + EPS)).map_err(|e| e.to_string())?;
    let norm = &xc / &denom;
    Ok(&(&norm * w) + b)
}

/// 멀티헤드 self-attention(인코더, 마스크/캐시 없음). x:[B,T,D].
fn self_attention(x: &Array, w: &Weights, p: &str, n_head: i32) -> Result<Array, String> {
    let s = x.shape();
    let (b, t, d) = (s[0], s[1], s[2]);
    let dh = d / n_head;
    let scale = (dh as f32).powf(-0.5);

    let q = linear(
        x,
        get(w, &format!("{p}.q_proj.weight"))?,
        w.get(&format!("{p}.q_proj.bias")),
    )?;
    let k = linear(
        x,
        get(w, &format!("{p}.k_proj.weight"))?,
        w.get(&format!("{p}.k_proj.bias")),
    )?;
    let v = linear(
        x,
        get(w, &format!("{p}.v_proj.weight"))?,
        w.get(&format!("{p}.v_proj.bias")),
    )?;

    let reshape_h = |a: &Array| -> Result<Array, String> {
        a.reshape(&[b, t, n_head, dh])
            .and_then(|r| r.transpose_axes(&[0, 2, 1, 3]))
            .map_err(|e| e.to_string())
    };
    let q = reshape_h(&q)?;
    let k = reshape_h(&k)?;
    let v = reshape_h(&v)?;

    let kt = k.transpose_axes(&[0, 1, 3, 2]).map_err(|e| e.to_string())?;
    let scores = q.matmul(&kt).map_err(|e| e.to_string())?;
    let scores = &scores * scale;
    let attn = ops::softmax_axis(&scores, -1, None).map_err(|e| e.to_string())?;
    let ctx = attn.matmul(&v).map_err(|e| e.to_string())?;
    let ctx = ctx
        .transpose_axes(&[0, 2, 1, 3])
        .and_then(|c| c.reshape(&[b, t, d]))
        .map_err(|e| e.to_string())?;
    linear(
        &ctx,
        get(w, &format!("{p}.out_proj.weight"))?,
        w.get(&format!("{p}.out_proj.bias")),
    )
}

/// 인코더 forward. mel:[1, n_mels, n_frames]. 반환 [1, n_audio_ctx, d_model].
pub fn encode(w: &Weights, mel: &Array, n_layers: i32, n_head: i32) -> Result<Array, String> {
    let conv1_w = get(w, "model.encoder.conv1.weight")?
        .transpose_axes(&[0, 2, 1])
        .map_err(|e| e.to_string())?;
    let conv2_w = get(w, "model.encoder.conv2.weight")?
        .transpose_axes(&[0, 2, 1])
        .map_err(|e| e.to_string())?;
    let x = mel.transpose_axes(&[0, 2, 1]).map_err(|e| e.to_string())?;

    let x = ops::conv1d(&x, &conv1_w, 1, 1, 1, 1).map_err(|e| format!("conv1: {e}"))?;
    let x = &x + get(w, "model.encoder.conv1.bias")?;
    let x = mlx_rs::nn::gelu(&x).map_err(|e| e.to_string())?;
    let x = ops::conv1d(&x, &conv2_w, 2, 1, 1, 1).map_err(|e| format!("conv2: {e}"))?;
    let x = &x + get(w, "model.encoder.conv2.bias")?;
    let mut x = mlx_rs::nn::gelu(&x).map_err(|e| e.to_string())?;

    let seq = x.shape()[1];
    let pos = get(w, "model.encoder.embed_positions.weight")?;
    let pos = pos.index((0..seq, ..));
    x = &x + &pos;

    for i in 0..n_layers {
        let p = format!("model.encoder.layers.{i}");
        let h = layer_norm(
            &x,
            get(w, &format!("{p}.self_attn_layer_norm.weight"))?,
            get(w, &format!("{p}.self_attn_layer_norm.bias"))?,
        )?;
        let attn = self_attention(&h, w, &format!("{p}.self_attn"), n_head)?;
        x = &x + &attn;
        let h = layer_norm(
            &x,
            get(w, &format!("{p}.final_layer_norm.weight"))?,
            get(w, &format!("{p}.final_layer_norm.bias"))?,
        )?;
        let h = linear(
            &h,
            get(w, &format!("{p}.fc1.weight"))?,
            w.get(&format!("{p}.fc1.bias")),
        )?;
        let h = mlx_rs::nn::gelu(&h).map_err(|e| e.to_string())?;
        let h = linear(
            &h,
            get(w, &format!("{p}.fc2.weight"))?,
            w.get(&format!("{p}.fc2.bias")),
        )?;
        x = &x + &h;
    }
    layer_norm(
        &x,
        get(w, "model.encoder.layer_norm.weight")?,
        get(w, "model.encoder.layer_norm.bias")?,
    )
}
