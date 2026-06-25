//! VAD 구현. `vad-energy` = 경량 RMS 기반(모델 불필요). asr-core 의 `Vad` trait 구현.
//! (신경망 Silero VAD 는 향후 vad-silero 로 추가 가능 — 현재는 energy 만.)

use asr_core::vad::Vad;

/// 경량 에너지(RMS) 기반 VAD. 의존성/모델/크래시 없이 무음을 걸러 ASR 추론을 절약한다.
/// 게이트는 추론만 건너뛰고 오디오는 버퍼에 남으므로 음성이 다시 감지되면 함께 전사됨(유실 없음).
pub struct EnergyVad {
    threshold: f32,
}

impl EnergyVad {
    /// threshold: 이 RMS 미만이면 무음으로 간주. 일반 발화 RMS≈0.01~0.1, 무음≈0.001~0.004.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Vad for EnergyVad {
    fn is_speech(&mut self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return false;
        }
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        rms >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_vad_gates_silence_passes_speech() {
        let mut v = EnergyVad::new(0.006);
        let silence = vec![0.0008f32; 1600]; // 무음/노이즈
        let speech: Vec<f32> = (0..1600)
            .map(|i| 0.05 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16_000.0).sin())
            .collect();
        assert!(!v.is_speech(&silence), "무음을 음성으로 오판");
        assert!(v.is_speech(&speech), "음성을 무음으로 오판");
    }
}
