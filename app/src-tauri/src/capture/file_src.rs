//! 파일 입력(데모/테스트 + 인코딩 파일 가져오기) — 오디오 파일을 16kHz mono 로 디코드해
//! **실시간 속도**로 파이프라인에 흘려보낸다. 마이크/시스템오디오 권한 없이도 전사가
//! 동작하는지 확인하는 용도이자, mp3/m4a 등 인코딩 파일을 가져오는 경로.
//!
//! 디코드는 Symphonia(순수 Rust)로 in-process 처리한다 — WhisperLiveKit 의 FFmpeg 서브프로세스
//! 방식은 iOS/Android 샌드박스에서 외부 바이너리를 띄울 수 없어 크로스플랫폼에 부적합하다.

use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::time::{Duration, Instant};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::resample::Resampler16k;
use super::{AudioFrame, AudioSource, CaptureControl};

/// 오디오 파일(mp3/m4a/flac/ogg/wav 등)을 mono f32 로 디코드해 (samples, sample_rate) 반환.
fn decode_to_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("열기 실패({path:?}): {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("오디오 포맷 인식 실패: {e}"))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "오디오 트랙이 없습니다".to_string())?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("디코더 생성 실패: {e}"))?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(16_000);
    let mut mono: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(format!("패킷 읽기 실패: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                let ch = spec.channels.count().max(1);
                let mut sb = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sb.copy_interleaved_ref(decoded);
                let samples = sb.samples();
                if ch <= 1 {
                    mono.extend_from_slice(samples);
                } else {
                    for frame in samples.chunks(ch) {
                        mono.push(frame.iter().sum::<f32>() / ch as f32);
                    }
                }
            }
            // 손상/부분 패킷은 건너뛴다(스트림 전체 실패로 키우지 않음).
            Err(SymError::DecodeError(_)) => continue,
            Err(SymError::IoError(ref e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("디코드 실패: {e}")),
        }
    }
    if mono.is_empty() {
        return Err(format!("디코드 결과가 비었습니다({path:?})"));
    }
    Ok((mono, sample_rate))
}

/// path 의 오디오를 16k mono 로 변환해 0.1초 청크씩 실시간 페이스로 tx 에 전송.
pub fn start_file(tx: Sender<AudioFrame>, path: PathBuf) -> Result<CaptureControl, String> {
    let (mono, sample_rate) = decode_to_mono(&path)?;
    // 16k 로 리샘플(필요 시)
    let samples: Vec<f32> = if sample_rate == 16_000 {
        mono
    } else {
        let mut rs = Resampler16k::new(sample_rate)?;
        let mut out = rs.push(&mono);
        out.extend(rs.push(&[])); // flush 잔여
        out
    };
    eprintln!(
        "[capture] 파일 입력 '{}' {:.1}s ({} samples @16k)",
        path.display(),
        samples.len() as f64 / 16_000.0,
        samples.len()
    );

    let (stop_tx, stop_rx) = channel::<()>();
    thread::spawn(move || {
        let chunk = 1600usize; // 0.1s
        let start = Instant::now();
        let mut t = 0.0f64;
        let mut i = 0usize;
        while i < samples.len() {
            if stop_rx.try_recv().is_ok() {
                break;
            }
            let end = (i + chunk).min(samples.len());
            let seg = samples[i..end].to_vec();
            let dur = seg.len() as f64 / 16_000.0;
            let t0 = t;
            t += dur;
            if tx
                .send(AudioFrame {
                    samples: seg,
                    t_start: t0,
                    t_end: t,
                    source: AudioSource::Mic,
                })
                .is_err()
            {
                break;
            }
            i = end;
            // 실시간 페이스: 재생시각 t 까지 경과를 맞춤.
            let target = Duration::from_secs_f64(t);
            let elapsed = start.elapsed();
            if target > elapsed {
                thread::sleep(target - elapsed);
            }
        }
    });
    Ok(CaptureControl::new(stop_tx))
}
