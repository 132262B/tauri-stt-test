//! cpal 마이크 입력 → mono 다운믹스 → 16kHz 리샘플 → `AudioFrame` mpsc 송신 (데스크톱).
//!
//! cpal `Stream` 은 macOS 에서 !Send 라 Tauri 상태에 직접 담을 수 없으므로,
//! 전용 스레드가 스트림을 소유한 채 stop 신호까지 블록한다. 신호가 오면 스트림이
//! 그 스레드에서 drop 되어 캡처가 멈춘다 (docs/02-architecture.md B.3).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc::{channel, Sender};
use std::thread;

use super::resample::{Resampler16k, TARGET_SR};
use super::{AudioFrame, AudioSource, CaptureControl};

/// 사용 가능한 입력 장치 이름 목록.
pub fn list_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devs) => devs.filter_map(|d| d.name().ok()).collect(),
        Err(_) => Vec::new(),
    }
}

/// 입력 장치에서 캡처를 시작한다. device_name=None 이면 기본 장치. 16kHz mono `AudioFrame` 송신.
pub fn start_mic(
    tx: Sender<AudioFrame>,
    device_name: Option<String>,
) -> Result<CaptureControl, String> {
    let (stop_tx, stop_rx) = channel::<()>();
    let (ready_tx, ready_rx) = channel::<Result<(), String>>();

    thread::spawn(move || match build_and_play(tx, device_name) {
        Ok(stream) => {
            ready_tx.send(Ok(())).ok();
            // stop 신호(또는 sender drop)까지 블록. 이후 stream 이 drop 되며 캡처 정지.
            let _ = stop_rx.recv();
            drop(stream);
        }
        Err(e) => {
            ready_tx.send(Err(e)).ok();
        }
    });

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(CaptureControl::new(stop_tx)),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("캡처 스레드 시작 실패".into()),
    }
}

fn build_and_play(
    tx: Sender<AudioFrame>,
    device_name: Option<String>,
) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = match device_name {
        Some(name) if !name.is_empty() => host
            .input_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| format!("입력 장치 '{name}' 를 찾을 수 없음"))?,
        _ => host
            .default_input_device()
            .ok_or("기본 입력 장치를 찾을 수 없음")?,
    };
    let dev_name = device.name().unwrap_or_else(|_| "?".into());
    let supported = device.default_input_config().map_err(|e| e.to_string())?;
    let in_sr = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let fmt = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();

    eprintln!("[capture] 입력장치='{dev_name}' sr={in_sr} ch={channels} fmt={fmt:?} → 16k mono");

    let mut resampler = Resampler16k::new(in_sr)?;
    let mut t_cursor = 0.0f64;

    // mono(16k 변환 전, 입력 sr) 샘플을 받아 리샘플·RMS 로깅·전송.
    let mut process_mono = move |mono: Vec<f32>| {
        let out = resampler.push(&mono);
        if out.is_empty() {
            return;
        }
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        let dur = out.len() as f64 / TARGET_SR as f64;
        let t_start = t_cursor;
        t_cursor += dur;
        if capture_verbose() {
            eprintln!(
                "[capture] 16k {}샘플 rms={rms:.4} t={t_start:.2}s",
                out.len()
            );
        }
        let _ = tx.send(AudioFrame {
            samples: out,
            t_start,
            t_end: t_cursor,
            source: AudioSource::Mic,
        });
    };

    let err_fn = |e| eprintln!("[capture] 스트림 오류: {e}");

    let stream = match fmt {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| process_mono(downmix(data, channels)),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                process_mono(downmix(&f, channels));
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _| {
                let f: Vec<f32> = data
                    .iter()
                    .map(|s| (*s as f32 - 32768.0) / 32768.0)
                    .collect();
                process_mono(downmix(&f, channels));
            },
            err_fn,
            None,
        ),
        other => return Err(format!("지원하지 않는 샘플 포맷: {other:?}")),
    }
    .map_err(|e| e.to_string())?;

    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

/// 인터리브드 멀티채널 → mono 평균.
fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        interleaved.to_vec()
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    }
}

fn capture_verbose() -> bool {
    std::env::var("ASR_CAPTURE_VERBOSE")
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
}
