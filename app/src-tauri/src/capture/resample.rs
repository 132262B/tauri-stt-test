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
