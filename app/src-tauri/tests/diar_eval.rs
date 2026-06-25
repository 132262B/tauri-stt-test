//! 화자분리 평가 — meeting.wav 를 전체 파이프라인(run_session, base+CAM++)에 흘려
//! 검출 화자 수(정답 4명: 박지민/정우성/한소희/김도현)와 라벨 전사를 점검한다.
//! 화자분리는 ASR 모델과 무관(CAM++ 임베딩)하므로 대표로 base 로 1회 측정.
//!
//! 실행: `cargo test -p app --test diar_eval -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use stt_core::asr::AsrConfig;
use stt_core::diar::Diarizer;
use stt_core::metrics::SessionMetrics;
use stt_core::output::TranscriptSnapshot;
use stt_core::pipeline::{run_session, AudioChunk};
use tokio::sync::mpsc;

const SR: usize = 16_000;
const OUT_DIR: &str = "/private/tmp/claude-501/-Users-kwonjunho-Desktop-work-tauri-stt-test/68e638b0-f4c7-4dbb-ba6f-f5c9ca7534ca/scratchpad/sweep";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "회의 음성 + 모델 필요. --ignored 로 실행"]
async fn diarization_eval() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wav = base.join("test-data/meeting.wav");
    assert!(wav.exists(), "meeting.wav 없음");
    std::fs::create_dir_all(OUT_DIR).ok();

    let mut r = hound::WavReader::open(&wav).expect("wav");
    let audio: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let audio_sec = audio.len() as f64 / SR as f64;

    let backend = stt_asr_whisper::self_streaming_backend(base.join("models/ggml"));
    let cfg = AsrConfig {
        model_id: "ggml-base".into(),
        language: Some("ko".into()),
        ..AsrConfig::default()
    };
    let diarizer: Option<Box<dyn Diarizer>> =
        match stt_diar::OnlineDiarizer::with_download(base.join("models/speaker"), 0.5) {
            Ok(d) => Some(Box::new(d)),
            Err(e) => panic!("화자분리 적재 실패: {e}"),
        };

    let (pcm_tx, pcm_rx) = mpsc::channel::<AudioChunk>(256);
    let (snap_tx, mut snap_rx) = mpsc::channel::<TranscriptSnapshot>(256);
    let reset = Arc::new(AtomicBool::new(false));

    // 1초 청크 피드(실시간 흐름 모사).
    let feeder = tokio::spawn(async move {
        let mut t = 0.0f64;
        let mut i = 0;
        while i < audio.len() {
            let end = (i + SR).min(audio.len());
            t += (end - i) as f64 / SR as f64;
            let _ = pcm_tx
                .send(AudioChunk { samples: audio[i..end].to_vec(), t_end: t })
                .await;
            i = end;
        }
        // drop pcm_tx → run_session 종료/flush
    });

    let driver = tokio::spawn(async move {
        let _ = run_session(
            backend,
            cfg,
            pcm_rx,
            snap_tx,
            SessionMetrics::default(),
            diarizer,
            None,
            reset,
        )
        .await;
    });

    let mut last: Option<TranscriptSnapshot> = None;
    while let Some(s) = snap_rx.recv().await {
        last = Some(s);
    }
    let _ = feeder.await;
    let _ = driver.await;

    let snap = last.expect("스냅샷 없음");

    // 화자 통계
    let mut by_spk: BTreeMap<i64, (usize, usize)> = BTreeMap::new(); // spk → (라인수, 글자수)
    let mut switches = 0usize;
    let mut prev: Option<u32> = None;
    let mut labeled = String::new();
    for l in &snap.lines {
        let key = l.speaker.map(|s| s as i64).unwrap_or(-1);
        let e = by_spk.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 += l.text.chars().count();
        if l.speaker != prev {
            switches += 1;
            prev = l.speaker;
        }
        let m = |t: f64| format!("{}:{:02}", (t as i64) / 60, (t as i64) % 60);
        labeled.push_str(&format!(
            "[화자 {}] {}~{}  {}\n",
            l.speaker.map(|s| (s + 1).to_string()).unwrap_or("?".into()),
            m(l.start),
            m(l.end),
            l.text.trim()
        ));
    }
    std::fs::write(format!("{OUT_DIR}/diarization.txt"), &labeled).ok();

    let n_spk = by_spk.keys().filter(|k| **k >= 0).count();
    eprintln!("\n============ 화자분리 평가 (base+CAM++, {audio_sec:.0}s) ============");
    eprintln!("정답 화자 수: 4 (박지민/정우성/한소희/김도현)");
    eprintln!("검출 화자 수: {n_spk}");
    eprintln!("총 라인 {}개, 화자 전환 {switches}회", snap.lines.len());
    eprintln!("화자별 분포(라인수, 글자수):");
    for (k, (lines, chars)) in &by_spk {
        let name = if *k < 0 { "미상".into() } else { format!("화자 {}", k + 1) };
        eprintln!("  {name}: {lines}라인 / {chars}자");
    }
    eprintln!("\n--- 라벨 전사 앞부분(20라인) ---");
    for line in labeled.lines().take(20) {
        eprintln!("{line}");
    }
    eprintln!("=================================================\n");
    eprintln!("전체 라벨 전사 저장: {OUT_DIR}/diarization.txt");
}
