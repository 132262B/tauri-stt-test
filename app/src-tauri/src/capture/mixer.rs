//! mic + system 두 16kHz mono 스트림을 가산 합성하여 단일 스트림으로 (C7 "둘 다").
//!
//! 두 소스는 각자 t_cursor 로 시작하므로 근사 정렬(둘 다 ~0 시작 가정)로 샘플 단위 합산.
//! 정밀 동기/에코 억제는 후속(docs/02-architecture.md B.6).
#![cfg(target_os = "macos")]

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use super::{AudioFrame, AudioSource};

const SR: f64 = 16_000.0;
const MIX_CHUNK: usize = 1600; // 100ms

/// mic_rx + sys_rx 를 합성해 out 으로 보내는 스레드를 띄운다. 두 소스가 모두 끝나면 종료.
pub fn spawn_mixer(mic_rx: Receiver<AudioFrame>, sys_rx: Receiver<AudioFrame>, out: Sender<AudioFrame>) {
    thread::spawn(move || {
        let mut mic: VecDeque<f32> = VecDeque::new();
        let mut sys: VecDeque<f32> = VecDeque::new();
        let mut mic_done = false;
        let mut sys_done = false;
        let mut t = 0.0_f64;

        loop {
            let mut got = false;
            loop {
                match mic_rx.try_recv() {
                    Ok(f) => {
                        mic.extend(f.samples);
                        got = true;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        mic_done = true;
                        break;
                    }
                }
            }
            loop {
                match sys_rx.try_recv() {
                    Ok(f) => {
                        sys.extend(f.samples);
                        got = true;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        sys_done = true;
                        break;
                    }
                }
            }

            let n = mic.len().min(sys.len());
            if n >= MIX_CHUNK {
                let mut mixed = Vec::with_capacity(n);
                for _ in 0..n {
                    let m = mic.pop_front().unwrap();
                    let s = sys.pop_front().unwrap();
                    mixed.push((m + s).clamp(-1.0, 1.0));
                }
                let dur = n as f64 / SR;
                let t_start = t;
                t += dur;
                if out
                    .send(AudioFrame {
                        samples: mixed,
                        t_start,
                        t_end: t,
                        source: AudioSource::System,
                    })
                    .is_err()
                {
                    break;
                }
            } else if mic_done && sys_done {
                break;
            } else if !got {
                thread::sleep(Duration::from_millis(10));
            }
        }
    });
}
