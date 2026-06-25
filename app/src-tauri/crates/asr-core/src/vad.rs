//! VAD(음성 활동 감지) **추상화 trait**. 무음 윈도우의 ASR 추론을 건너뛰어 연산을 절약.
//! 구현체는 별도 크레이트(`vad-energy` = vad-energy). diar 와 동일한 trait/구현 분리 패턴.

pub trait Vad: Send {
    /// 16kHz mono 샘플을 받아 현재 음성 활동 여부를 반환.
    fn is_speech(&mut self, samples: &[f32]) -> bool;
}
