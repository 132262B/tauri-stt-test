//! VAD(음성 활동 감지) 추상화 (docs/02-architecture.md, Seed C4 — whisper-live-kit Silero 참고).
//! 구현은 stt-diar(sherpa-onnx Silero). 무음 윈도우의 ASR 추론을 건너뛰어 연산을 절약.

pub trait Vad: Send {
    /// 16kHz mono 샘플을 받아 현재 음성 활동 여부를 반환.
    fn is_speech(&mut self, samples: &[f32]) -> bool;
}

/// 경량 에너지(RMS) 기반 VAD. sherpa Silero 가 SIGSEGV 라 의존성/크래시 없이 무음을
/// 걸러 추론을 절약한다. 게이트는 process_iter 만 건너뛰고 오디오는 버퍼에 남으므로
/// 음성이 다시 감지될 때 함께 전사된다(유실 없음).
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
