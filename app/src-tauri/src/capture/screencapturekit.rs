//! macOS 시스템 오디오 캡처(C7) — ScreenCaptureKit. 16kHz mono f32 `AudioFrame` 생성.
//!
//! 화면 녹화/시스템 오디오 권한(System Settings → Privacy → Screen & System Audio Recording)이
//! 필요하다. 첫 `SCShareableContent::get()` 시 권한 프롬프트가 뜬다.
#![cfg(target_os = "macos")]

use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::thread;

use screencapturekit::cm::AudioBufferList;
use screencapturekit::prelude::*;

use super::{AudioFrame, AudioSource, CaptureControl};

const SR: f64 = 16_000.0;

#[derive(Clone, Copy, Debug)]
struct AudioFormat {
    sample_rate: f64,
    channels: usize,
    bits_per_channel: Option<u32>,
    bytes_per_frame: Option<u32>,
    is_float: Option<bool>,
    is_big_endian: bool,
}

#[derive(Clone, Copy, Debug)]
enum SampleEncoding {
    F32,
    I16,
}

impl SampleEncoding {
    fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I16 => "i16",
        }
    }
}

/// SCK 오디오 콜백 핸들러. SCStreamOutputTrait 은 Send+Sync 필요 → Mutex 로 감쌈.
struct AudioHandler {
    tx: Mutex<Sender<AudioFrame>>,
    t_cursor: Mutex<f64>,
    last_log_sec: Mutex<u64>,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type != SCStreamOutputType::Audio {
            return;
        }
        let _ = sample.make_data_ready();
        let Some(list) = sample.audio_buffer_list() else {
            return;
        };
        let Some(buf) = list.get(0) else { return };
        let format = audio_format(&sample, buf.number_channels.max(1) as usize);
        let (mono, encoding) = decode_audio_buffer_list(&list, &format);
        if mono.is_empty() {
            return;
        }
        let decoded_rms = rms(&mono);
        let resampled = resample_to_16k(&mono, format.sample_rate);
        if resampled.is_empty() {
            return;
        }
        let raw_rms = rms(&resampled);
        let gain = system_audio_gain(raw_rms);
        let samples: Vec<f32> = if gain > 1.0 {
            resampled
                .iter()
                .map(|s| (s * gain).clamp(-1.0, 1.0))
                .collect()
        } else {
            resampled
        };
        let out_rms = if gain > 1.0 { rms(&samples) } else { raw_rms };

        let dur = samples.len() as f64 / SR;
        let (t_start, t_end) = {
            let mut c = self.t_cursor.lock().unwrap();
            let s = *c;
            *c += dur;
            (s, *c)
        };
        if let Ok(mut last) = self.last_log_sec.lock() {
            let sec = t_end as u64;
            if sec != *last {
                *last = sec;
                let total_bytes: usize = list.iter().map(|b| b.data_byte_size()).sum();
                let buffers = list.num_buffers();
                let bits = format
                    .bits_per_channel
                    .map_or_else(|| "?".to_string(), |v| v.to_string());
                let bpf = format
                    .bytes_per_frame
                    .map_or_else(|| "?".to_string(), |v| v.to_string());
                let float = format
                    .is_float
                    .map_or_else(|| "?".to_string(), |v| v.to_string());
                eprintln!(
                    "[capture:system] sr={:.0}Hz ch={} buffers={} bytes={} enc={} bits={} bpf={} float={} decoded_rms={decoded_rms:.6} raw_rms={raw_rms:.6} gain={gain:.1} out_rms={out_rms:.6} t={t_end:.1}s",
                    format.sample_rate,
                    format.channels,
                    buffers,
                    total_bytes,
                    encoding.label(),
                    bits,
                    bpf,
                    float,
                );
            }
        }
        if let Ok(tx) = self.tx.lock() {
            let _ = tx.send(AudioFrame {
                samples,
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
            last_log_sec: Mutex::new(0),
        },
        SCStreamOutputType::Audio,
    );
    stream
        .start_capture()
        .map_err(|e| format!("시스템 오디오 캡처 시작 실패: {e}"))?;
    Ok(stream)
}

fn audio_format(sample: &CMSampleBuffer, fallback_channels: usize) -> AudioFormat {
    let desc = sample.format_description();
    let channels = desc
        .as_ref()
        .and_then(|d| d.audio_channel_count())
        .map(|v| v as usize)
        .unwrap_or(fallback_channels)
        .max(1);
    AudioFormat {
        sample_rate: desc
            .as_ref()
            .and_then(|d| d.audio_sample_rate())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(SR),
        channels,
        bits_per_channel: desc.as_ref().and_then(|d| d.audio_bits_per_channel()),
        bytes_per_frame: desc.as_ref().and_then(|d| d.audio_bytes_per_frame()),
        is_float: desc.as_ref().map(|d| d.audio_is_float()),
        is_big_endian: desc.as_ref().is_some_and(|d| d.audio_is_big_endian()),
    }
}

fn decode_audio_buffer_list(
    list: &AudioBufferList,
    format: &AudioFormat,
) -> (Vec<f32>, SampleEncoding) {
    if list.num_buffers() <= 1 {
        let Some(buf) = list.get(0) else {
            return (Vec::new(), SampleEncoding::F32);
        };
        let channels = (buf.number_channels as usize).max(format.channels).max(1);
        return decode_interleaved(buf.data(), channels, format);
    }

    let mut tracks = Vec::new();
    let mut encoding = SampleEncoding::F32;
    for buf in list.iter() {
        let (track, enc) = decode_interleaved(buf.data(), 1, format);
        if !track.is_empty() {
            tracks.push(track);
            encoding = enc;
        }
    }
    let Some(min_len) = tracks.iter().map(Vec::len).min() else {
        return (Vec::new(), encoding);
    };
    let mut mono = vec![0.0f32; min_len];
    for track in &tracks {
        for (dst, src) in mono.iter_mut().zip(track.iter()) {
            *dst += *src;
        }
    }
    let scale = tracks.len() as f32;
    for sample in &mut mono {
        *sample /= scale;
    }
    (mono, encoding)
}

fn decode_interleaved(
    bytes: &[u8],
    channels: usize,
    format: &AudioFormat,
) -> (Vec<f32>, SampleEncoding) {
    let primary = choose_encoding(format, channels);
    let samples = decode_with_encoding(bytes, channels, primary, format.is_big_endian);
    if !samples.is_empty() {
        return (samples, primary);
    }
    let fallback = match primary {
        SampleEncoding::F32 => SampleEncoding::I16,
        SampleEncoding::I16 => SampleEncoding::F32,
    };
    (
        decode_with_encoding(bytes, channels, fallback, format.is_big_endian),
        fallback,
    )
}

fn choose_encoding(format: &AudioFormat, channels: usize) -> SampleEncoding {
    if format.is_float == Some(true) || format.bits_per_channel == Some(32) {
        return SampleEncoding::F32;
    }
    if format.bits_per_channel == Some(16) {
        return SampleEncoding::I16;
    }
    match format.bytes_per_frame {
        Some(bpf) if bpf == (channels * 2) as u32 => SampleEncoding::I16,
        _ => SampleEncoding::F32,
    }
}

fn decode_with_encoding(
    bytes: &[u8],
    channels: usize,
    encoding: SampleEncoding,
    big_endian: bool,
) -> Vec<f32> {
    match encoding {
        SampleEncoding::F32 => decode_f32(bytes, channels, big_endian),
        SampleEncoding::I16 => decode_i16(bytes, channels, big_endian),
    }
}

fn decode_f32(bytes: &[u8], channels: usize, big_endian: bool) -> Vec<f32> {
    let frame_bytes = channels.saturating_mul(4);
    if frame_bytes == 0 || bytes.len() < frame_bytes {
        return Vec::new();
    }
    bytes
        .chunks_exact(frame_bytes)
        .filter_map(|frame| {
            let mut sum = 0.0;
            let mut used = 0usize;
            for ch in 0..channels {
                let offset = ch * 4;
                let raw = [
                    frame[offset],
                    frame[offset + 1],
                    frame[offset + 2],
                    frame[offset + 3],
                ];
                let value = if big_endian {
                    f32::from_be_bytes(raw)
                } else {
                    f32::from_le_bytes(raw)
                };
                if value.is_finite() && value.abs() <= 8.0 {
                    sum += value.clamp(-1.0, 1.0);
                    used += 1;
                }
            }
            (used > 0).then_some(sum / used as f32)
        })
        .collect()
}

fn decode_i16(bytes: &[u8], channels: usize, big_endian: bool) -> Vec<f32> {
    let frame_bytes = channels.saturating_mul(2);
    if frame_bytes == 0 || bytes.len() < frame_bytes {
        return Vec::new();
    }
    bytes
        .chunks_exact(frame_bytes)
        .map(|frame| {
            let mut sum = 0.0;
            for ch in 0..channels {
                let offset = ch * 2;
                let raw = [frame[offset], frame[offset + 1]];
                let value = if big_endian {
                    i16::from_be_bytes(raw)
                } else {
                    i16::from_le_bytes(raw)
                };
                sum += value as f32 / i16::MAX as f32;
            }
            (sum / channels as f32).clamp(-1.0, 1.0)
        })
        .collect()
}

fn resample_to_16k(samples: &[f32], input_rate: f64) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if !input_rate.is_finite() || input_rate <= 0.0 || (input_rate - SR).abs() < 1.0 {
        return samples.to_vec();
    }

    let out_len = ((samples.len() as f64) * SR / input_rate).round().max(1.0) as usize;
    let step = input_rate / SR;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * step;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = samples.get(idx).copied().unwrap_or(0.0);
        let b = samples.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn system_audio_gain(raw_rms: f32) -> f32 {
    if raw_rms < 0.00005 {
        return 1.0;
    }
    let target = 0.03;
    (target / raw_rms).clamp(1.0, 20.0)
}
