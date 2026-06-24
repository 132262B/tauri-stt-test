use std::time::Instant;

use tokio::sync::mpsc;

use crate::asr::{AsrConfig, AsrError, AsrToken, StreamingAsrBackend};
use crate::metrics::SessionMetrics;
use crate::output::{CommittedToken, TranscriptLine, TranscriptSnapshot};

const SAMPLE_RATE: usize = 16_000;
/// 이만큼(샘플) 누적될 때마다 process_iter 1회(≈1초).
const ITER_SAMPLES: usize = SAMPLE_RATE;
/// 같은 화자라도 이보다 큰 공백이면 라인을 끊는다(초).
const LINE_GAP_SEC: f64 = 3.0;

/// 캡처가 공급하는 16kHz mono PCM 청크(절대 끝시각 포함).
#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub t_end: f64,
}

/// 세션 드라이버: PCM 을 백엔드에 공급하고 주기적으로 추론하여 전사 스냅샷을 emit.
///
/// 확정 토큰을 화자별 라인으로 묶어 누적한다. pcm_rx 가 닫히면 마지막 flush 후 종료.
pub async fn run_session(
    mut backend: Box<dyn StreamingAsrBackend>,
    cfg: AsrConfig,
    mut pcm_rx: mpsc::Receiver<AudioChunk>,
    snap_tx: mpsc::Sender<TranscriptSnapshot>,
    metrics: SessionMetrics,
) -> Result<(), AsrError> {
    backend.configure(&cfg).await?;
    let _ = backend.warmup().await;

    let mut committed_text = String::new();
    let mut lines: Vec<TranscriptLine> = Vec::new();
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
            append_committed(&mut lines, &mut committed_text, &committed);
            let _ = snap_tx
                .send(TranscriptSnapshot {
                    committed_text: committed_text.clone(),
                    lines: lines.clone(),
                    buffer: backend.get_buffer(),
                    buffer_speaker: None,
                    upto: last_upto,
                    new_committed: to_committed(&committed),
                })
                .await;
        }
    }

    // 최종 flush
    let remaining = backend.process_iter(true).await?;
    append_committed(&mut lines, &mut committed_text, &remaining);
    let _ = snap_tx
        .send(TranscriptSnapshot {
            committed_text,
            lines,
            buffer: String::new(),
            buffer_speaker: None,
            upto: last_upto,
            new_committed: to_committed(&remaining),
        })
        .await;
    Ok(())
}

/// 확정 토큰 배치를 화자별 라인으로 누적(같은 화자+근접 시각이면 기존 라인 연장).
fn append_committed(lines: &mut Vec<TranscriptLine>, committed_text: &mut String, batch: &[AsrToken]) {
    if batch.is_empty() {
        return;
    }
    let spk = batch[0].speaker;
    let start = batch[0].start;
    let end = batch[batch.len() - 1].end;
    let text: String = batch.iter().map(|t| t.text.as_str()).collect();
    committed_text.push_str(&text);

    let extend = lines
        .last()
        .map(|l| l.speaker == spk && (start - l.end) < LINE_GAP_SEC)
        .unwrap_or(false);
    if extend {
        let last = lines.last_mut().unwrap();
        last.text.push_str(&text);
        last.end = end;
    } else {
        lines.push(TranscriptLine {
            speaker: spk,
            text,
            start,
            end,
        });
    }
}

fn to_committed(tokens: &[AsrToken]) -> Vec<CommittedToken> {
    tokens
        .iter()
        .map(|t| CommittedToken {
            start: t.start,
            end: t.end,
            text: t.text.clone(),
            speaker: t.speaker,
        })
        .collect()
}
