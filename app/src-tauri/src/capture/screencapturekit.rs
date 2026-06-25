//! macOS 시스템 오디오 캡처(C7) — ScreenCaptureKit. 16kHz mono f32 `AudioFrame` 생성.
//!
//! 화면 녹화/시스템 오디오 권한(System Settings → Privacy → Screen & System Audio Recording)이
//! 필요하다. 첫 `SCShareableContent::get()` 시 권한 프롬프트가 뜬다.
#![cfg(target_os = "macos")]

use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::thread;

use screencapturekit::prelude::*;

use super::{AudioFrame, AudioSource, CaptureControl};

const SR: f64 = 16_000.0;

/// SCK 오디오 콜백 핸들러. SCStreamOutputTrait 은 Send+Sync 필요 → Mutex 로 감쌈.
struct AudioHandler {
    tx: Mutex<Sender<AudioFrame>>,
    t_cursor: Mutex<f64>,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type != SCStreamOutputType::Audio {
            return;
        }
        let _ = sample.make_data_ready();
        let Some(list) = sample.audio_buffer_list() else { return };
        let Some(buf) = list.get(0) else { return };
        let channels = buf.number_channels.max(1) as usize;

        // f32 LE PCM
        let bytes = buf.data();
        let interleaved: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mono: Vec<f32> = if channels <= 1 {
            interleaved
        } else {
            interleaved
                .chunks(channels)
                .map(|f| f.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        if mono.is_empty() {
            return;
        }

        let dur = mono.len() as f64 / SR;
        let (t_start, t_end) = {
            let mut c = self.t_cursor.lock().unwrap();
            let s = *c;
            *c += dur;
            (s, *c)
        };
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(AudioFrame {
                samples: mono,
                t_start,
                t_end,
                source: AudioSource::System,
            });
        }
    }
}

/// 시스템 오디오 캡처 시작. 16kHz mono `AudioFrame` 을 tx 로 보낸다.
pub fn start_system(tx: Sender<AudioFrame>) -> Result<CaptureControl, String> {
    let (stop_tx, stop_rx) = channel::<()>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();

    // SCK 셋업은 권한·콘텐츠 조회가 블록될 수 있어 전용 스레드에서. 스트림은 그 스레드가 소유.
    thread::spawn(move || match build_and_start(tx) {
        Ok(stream) => {
            ready_tx.send(Ok(())).ok();
            let _ = stop_rx.recv();
            let _ = stream.stop_capture();
        }
        Err(e) => {
            ready_tx.send(Err(e)).ok();
        }
    });

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(CaptureControl::new(stop_tx)),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("시스템 오디오 스레드 시작 실패".into()),
    }
}

fn build_and_start(tx: Sender<AudioFrame>) -> Result<SCStream, String> {
    let content = SCShareableContent::get()
        .map_err(|e| format!("화면 공유 콘텐츠 조회 실패(화면 녹화 권한 필요): {e}"))?;
    let display = content
        .displays()
        .into_iter()
        .next()
        .ok_or("디스플레이를 찾을 수 없음")?;

    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build();

    let config = SCStreamConfiguration::new()
        .with_captures_audio(true)
        .with_sample_rate(16000)
        .with_channel_count(1);

    let mut stream = SCStream::new(&filter, &config);
    stream.add_output_handler(
        AudioHandler {
            tx: Mutex::new(tx),
            t_cursor: Mutex::new(0.0),
        },
        SCStreamOutputType::Audio,
    );
    stream
        .start_capture()
        .map_err(|e| format!("시스템 오디오 캡처 시작 실패: {e}"))?;
    Ok(stream)
}
