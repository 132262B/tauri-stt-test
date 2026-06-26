//! 임의 입력 샘플레이트의 mono f32 → 16kHz mono f32 리샘플 (rubato FFT, anti-alias).
//!
//! cpal 콜백은 가변 길이 버퍼를 주므로, 입력을 고정 청크로 누적해 리샘플러에 공급한다.

use rubato::{FftFixedIn, Resampler};

pub const TARGET_SR: u32 = 16_000;

pub struct Resampler16k {
    inner: FftFixedIn<f32>,
    in_buf: Vec<f32>,
    chunk: usize,
}

impl Resampler16k {
    pub fn new(input_sr: u32) -> Result<Self, String> {
        let chunk = 1024usize;
        let inner = FftFixedIn::<f32>::new(input_sr as usize, TARGET_SR as usize, chunk, 2, 1)
            .map_err(|e| format!("리샘플러 초기화 실패: {e}"))?;
        Ok(Self {
            inner,
            in_buf: Vec::new(),
            chunk,
        })
    }

    /// mono 입력 샘플을 누적하고, 고정 청크가 모이는 대로 16kHz mono 출력을 반환한다.
    pub fn push(&mut self, mono_in: &[f32]) -> Vec<f32> {
        self.in_buf.extend_from_slice(mono_in);
        let mut out = Vec::new();
        while self.in_buf.len() >= self.chunk {
            let frame: Vec<f32> = self.in_buf.drain(..self.chunk).collect();
            match self.inner.process(&[frame], None) {
                Ok(mut res) => {
                    if let Some(ch0) = res.drain(..).next() {
                        out.extend(ch0);
                    }
                }
                Err(e) => {
                    eprintln!("[capture] 리샘플 오류: {e}");
                    break;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 알려진 진폭의 48k 사인파 → 16k 리샘플 후 RMS 가 보존되는지(캡처 경로 감쇠 진단).
    #[test]
    fn resample_preserves_amplitude() {
        let mut r = Resampler16k::new(48_000).expect("resampler");
        let n = 48_000usize; // 1s @48k
        let amp = 0.5f32;
        let input: Vec<f32> = (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let in_rms = (input.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        let out = r.push(&input);
        assert!(!out.is_empty(), "출력 없음");
        let out_rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        eprintln!(
            "in_rms={in_rms:.4} out_rms={out_rms:.4} (n_out={})",
            out.len()
        );
        // 사인파 rms ≈ amp/√2 ≈ 0.354. 리샘플 후 ±20% 이내면 감쇠 없음.
        assert!(
            (out_rms - in_rms).abs() < in_rms * 0.2,
            "리샘플러가 진폭을 크게 바꿈: in={in_rms} out={out_rms}"
        );
    }
}
