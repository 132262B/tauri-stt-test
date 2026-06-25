use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::asr::{AsrConfig, AsrError, AsrToken, StreamingAsrBackend};
use crate::diar::{DiarSegment, Diarizer};
use crate::metrics::SessionMetrics;
use crate::output::{CommittedToken, TranscriptLine, TranscriptSnapshot};
use crate::vad::Vad;

const SAMPLE_RATE: usize = 16_000;
/// 같은 화자라도 이보다 큰 공백이면 라인을 끊는다(초).
const LINE_GAP_SEC: f64 = 1.5;

/// 캡처가 공급하는 16kHz mono PCM 청크(절대 끝시각 포함).
#[derive(Clone, Debug)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub t_end: f64,
}

/// 세션 드라이버: PCM 을 백엔드에 공급하고 주기적으로 추론해 전사 스냅샷을 emit.
///
/// 화자분리는 **백그라운드 스레드**에서 누적 오디오 전체를 주기적으로 분석(pyannote, 오프라인)하고,
/// 메인 루프는 매 emit 시 최신 세그먼트로 토큰을 화자에 매핑한다(전사를 막지 않음).
pub async fn run_session(
    mut backend: Box<dyn StreamingAsrBackend>,
    cfg: AsrConfig,
    mut pcm_rx: mpsc::Receiver<AudioChunk>,
    snap_tx: mpsc::Sender<TranscriptSnapshot>,
    metrics: SessionMetrics,
    diarizer: Option<Box<dyn Diarizer>>,
    mut vad: Option<Box<dyn Vad>>,
    reset: Arc<AtomicBool>,
) -> Result<(), AsrError> {
    backend.configure(&cfg).await?;
    let _ = backend.warmup().await;

    let m = cfg.model_id.to_lowercase();
    let heavy = m.contains("large") || m.contains("turbo") || m.contains("qwen") || m == "sensevoice";
    let iter_samples = if heavy { 2 * SAMPLE_RATE } else { SAMPLE_RATE };

    // 화자분리는 **정지(finalize) 시 1회만** 수행한다. 라이브 중 백그라운드로 돌리면
    // 8스레드 ONNX 가 CPU 를 점유해 전사까지 느려지고 발열/팬이 심해지므로(코어 경합),
    // 라이브엔 전사만 돌리고(빠름), 종료 시 누적 오디오 전체를 한 번 분석해 라벨링한다.
    let has_diar = diarizer.is_some();

    let mut committed_text = String::new();
    let mut all_tokens: Vec<AsrToken> = Vec::new();
    let mut full_audio: Vec<f32> = Vec::new();
    let segments: Vec<DiarSegment> = Vec::new(); // 라이브 중엔 비어있음(라벨 None)
    let mut since_iter = 0usize;
    let mut speech_in_window = false;
    let mut last_upto = 0.0_f64;

    while let Some(chunk) = pcm_rx.recv().await {
        if reset.swap(false, Ordering::Relaxed) {
            committed_text.clear();
            all_tokens.clear();
            full_audio.clear();
            let _ = snap_tx.send(empty_snapshot(last_upto)).await;
        }
        last_upto = chunk.t_end;
        since_iter += chunk.samples.len();
        match vad.as_mut() {
            Some(v) => {
                if v.is_speech(&chunk.samples) {
                    speech_in_window = true;
                }
            }
            None => speech_in_window = true,
        }
        backend.insert_audio_chunk(&chunk.samples, chunk.t_end);
        if has_diar {
            full_audio.extend_from_slice(&chunk.samples);
        }

        if since_iter >= iter_samples {
            if !speech_in_window {
                since_iter = 0;
                continue;
            }
            speech_in_window = false;
            let audio_sec = since_iter as f32 / SAMPLE_RATE as f32;
            since_iter = 0;
            let t0 = Instant::now();
            let committed = backend.process_iter(false).await?;
            metrics.record_iter(t0.elapsed().as_secs_f32() * 1000.0, audio_sec);
            all_tokens.extend(committed.iter().cloned());
            for t in &committed {
                committed_text.push_str(&t.text);
            }
            let lines = build_lines(&all_tokens, &segments);
            let _ = snap_tx
                .send(TranscriptSnapshot {
                    committed_text: committed_text.clone(),
                    lines,
                    buffer: backend.get_buffer(),
                    buffer_speaker: None,
                    upto: last_upto,
                    new_committed: to_committed(&committed),
                })
                .await;
        }
    }

    // 최종 flush.
    let remaining = backend.process_iter(true).await?;
    all_tokens.extend(remaining.iter().cloned());
    for t in &remaining {
        committed_text.push_str(&t.text);
    }
    // === finalize 화자분리: 종료 시 누적 오디오 전체를 1회 분석(전사 끝나서 CPU 경합 없음). ===
    let mut segments = segments;
    if let Some(mut d) = diarizer {
        let secs = full_audio.len() as f64 / SAMPLE_RATE as f64;
        let t0 = Instant::now();
        segments = d.diarize(&full_audio);
        eprintln!(
            "[diar] (정지 시) 오디오 {:.0}s → {}세그먼트, {:.0}ms",
            secs,
            segments.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    let lines = build_lines(&all_tokens, &segments);
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

fn empty_snapshot(upto: f64) -> TranscriptSnapshot {
    TranscriptSnapshot {
        committed_text: String::new(),
        lines: Vec::new(),
        buffer: String::new(),
        buffer_speaker: None,
        upto,
        new_committed: Vec::new(),
    }
}

/// 토큰의 중간시각이 속한 diar 세그먼트의 화자(겹침 최대). 세그먼트 없으면 None.
fn speaker_at(segments: &[DiarSegment], start: f64, end: f64) -> Option<u32> {
    if segments.is_empty() {
        return None;
    }
    let mid = (start + end) / 2.0;
    if let Some(s) = segments.iter().find(|s| s.start <= mid && mid <= s.end) {
        return Some(s.speaker);
    }
    segments
        .iter()
        .map(|s| (s.speaker, (end.min(s.end) - start.max(s.start)).max(0.0)))
        .filter(|(_, ov)| *ov > 0.0)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(spk, _)| spk)
}

/// 전체 토큰 + 최신 diar 세그먼트로 화자별 라인 구성(화자 바뀌거나 공백 크면 분리).
fn build_lines(tokens: &[AsrToken], segments: &[DiarSegment]) -> Vec<TranscriptLine> {
    let mut lines: Vec<TranscriptLine> = Vec::new();
    for t in tokens {
        let spk = speaker_at(segments, t.start, t.end);
        let extend = lines
            .last()
            .map(|l| l.speaker == spk && (t.start - l.end) < LINE_GAP_SEC)
            .unwrap_or(false);
        if extend {
            let last = lines.last_mut().unwrap();
            last.text.push_str(&t.text);
            last.end = t.end;
        } else {
            lines.push(TranscriptLine {
                speaker: spk,
                text: t.text.clone(),
                start: t.start,
                end: t.end,
            });
        }
    }
    lines
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
