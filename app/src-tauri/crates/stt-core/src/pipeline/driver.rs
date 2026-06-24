use std::time::Instant;

use tokio::sync::mpsc;

use crate::asr::{AsrConfig, AsrError, StreamingAsrBackend};
use crate::metrics::SessionMetrics;
use crate::output::TranscriptSnapshot;

const SAMPLE_RATE: usize = 16_000;
/// 이만큼(샘플) 누적될 때마다 process_iter 1회(≈1초).
const ITER_SAMPLES: usize = SAMPLE_RATE;

/// 캡처가 공급하는 16kHz mono PCM 청크(절대 끝시각 포함).
#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub t_end: f64,
}

/// 세션 드라이버: PCM 을 백엔드에 공급하고 주기적으로 추론하여 전사 스냅샷을 emit.
///
/// pcm_rx 가 닫히면(캡처 종료) 마지막 flush 후 종료. tauri/플랫폼 무관.
pub async fn run_session(
    mut backend: Box<dyn StreamingAsrBackend>,
    cfg: AsrConfig,
    mut pcm_rx: mpsc::Receiver<AudioChunk>,
    snap_tx: mpsc::Sender<TranscriptSnapshot>,
    metrics: SessionMetrics,
) -> Result<(), AsrError> {
    backend.configure(&cfg).await?;
    // 예열 실패는 치명적이지 않음(모델은 configure 에서 적재됨).
    let _ = backend.warmup().await;

    let mut committed_text = String::new();
    let mut since_iter = 0usize;
    let mut last_upto = 0.0_f64;

    while let Some(chunk) = pcm_rx.recv().await {
        last_upto = chunk.t_end;
        since_iter += chunk.samples.len();
        backend.insert_audio_chunk(&chunk.samples, chunk.t_end);

        if since_iter >= ITER_SAMPLES {
            let audio_sec = since_iter as f32 / SAMPLE_RATE as f32;
            since_iter = 0;
            let t0 = Instant::now();
            let committed = backend.process_iter(false).await?;
            metrics.record_iter(t0.elapsed().as_secs_f32() * 1000.0, audio_sec);
            for t in &committed {
                committed_text.push_str(&t.text);
            }
            let _ = snap_tx
                .send(TranscriptSnapshot {
                    committed_text: committed_text.clone(),
                    buffer: backend.get_buffer(),
                    upto: last_upto,
                })
                .await;
        }
    }

    // 최종 flush
    let remaining = backend.process_iter(true).await?;
    for t in &remaining {
        committed_text.push_str(&t.text);
    }
    let _ = snap_tx
        .send(TranscriptSnapshot {
            committed_text,
            buffer: String::new(),
            upto: last_upto,
        })
        .await;
    Ok(())
}
