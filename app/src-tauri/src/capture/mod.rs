//! 플랫폼 의존 오디오 캡처 (docs/02-architecture.md A.1 capture/).
//!
//! 캡처는 16kHz mono f32 PCM `AudioFrame` 을 생성해 파이프라인 mpsc 로 보낸다.
//! P1 커밋4: 마이크(cpal) 입력만. 시스템 오디오(ScreenCaptureKit)·믹서는 P1.5,
//! iOS 마이크(AVAudioEngine)는 P3.

pub mod resample;

#[cfg(desktop)]
pub mod mic_cpal;

/// 캡처 소스에서 나오는 16kHz mono f32 PCM 프레임(절대시각 메타 포함).
// 필드는 커밋9(파이프라인 연결)에서 소비된다.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    pub t_start: f64,
    pub t_end: f64,
    pub source: AudioSource,
}

// System 은 P1.5(ScreenCaptureKit 시스템오디오)에서 생성된다.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioSource {
    Mic,
    System,
}

/// 실행 중인 캡처를 정지시키는 핸들. `stop()` 시 캡처 스레드가 종료되며
/// cpal 스트림(!Send)이 그 스레드에서 drop 되어 캡처가 멈춘다.
pub struct CaptureControl {
    stop_tx: std::sync::mpsc::Sender<()>,
}

impl CaptureControl {
    pub fn new(stop_tx: std::sync::mpsc::Sender<()>) -> Self {
        Self { stop_tx }
    }

    pub fn stop(self) {
        let _ = self.stop_tx.send(());
    }
}
