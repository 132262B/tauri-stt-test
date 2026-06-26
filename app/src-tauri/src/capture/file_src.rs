//! 파일 입력(데모/테스트용) — WAV 를 16kHz mono 로 읽어 **실시간 속도**로 파이프라인에
//! 흘려보낸다. 마이크/시스템오디오 권한 없이도 전사가 동작하는지 앱에서 직접 확인하는 용도.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::time::{Duration, Instant};

use super::resample::Resampler16k;
use super::{AudioFrame, AudioSource, CaptureControl};

/// path 의 WAV 를 16k mono 로 변환해 0.1초 청크씩 실시간 페이스로 tx 에 전송.
pub fn start_file(tx: Sender<AudioFrame>, path: PathBuf) -> Result<CaptureControl, String> {
    let mut reader =
        hound::WavReader::open(&path).map_err(|e| format!("WAV 열기 실패({path:?}): {e}"))?;
    let spec = reader.spec();
    let ch = spec.channels as usize;
    // 원본 샘플(인터리브) → mono f32
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
    };
    let mono: Vec<f32> = if ch <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    // 16k 로 리샘플(필요 시)
    let samples: Vec<f32> = if spec.sample_rate == 16_000 {
        mono
    } else {
        let mut rs = Resampler16k::new(spec.sample_rate)?;
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
