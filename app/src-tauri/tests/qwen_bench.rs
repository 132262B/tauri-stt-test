//! Qwen3-ASR 정확도 측정 — 긴 회의 음성을 30초 청크로 분할 전사(단일패스는 컨텍스트
//! 폭발로 비현실적). 전사를 저장해 CER 분석에 사용. 처리시간도 기록.
//!
//! 실행: `cargo test -p app --test qwen_bench -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use asr_qwen::{QwenBackend, QWEN_06B, QWEN_17B};
use asr_core::asr::SelfStreamingBackend;

const SR: usize = 16_000;
const OUT_DIR: &str = "/private/tmp/claude-501/-Users-kwonjunho-Desktop-work-tauri-stt-test/68e638b0-f4c7-4dbb-ba6f-f5c9ca7534ca/scratchpad/sweep";

#[test]
#[ignore = "Qwen 모델 + 회의 음성 필요. --ignored 로 실행"]
fn qwen_chunked_bench() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut r = hound::WavReader::open(base.join("test-data/meeting.wav")).expect("wav");
    let audio: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let audio_sec = audio.len() as f64 / SR as f64;
    std::fs::create_dir_all(OUT_DIR).ok();

    let models: &[(&str, &str, &asr_qwen::QwenModelSpec)] = &[
        ("qwen-0.6b", "models/qwen", &QWEN_06B),
        ("qwen-1.7b", "models/qwen-1.7b", &QWEN_17B),
    ];
    let chunk = 30 * SR; // 30초 청크(발화 경계 근처에서 분할 → 품질 유지 + 속도 정상)

    for (name, dir, spec) in models {
        let d = base.join(dir);
        if !d.join("config.json").exists() {
            eprintln!("[skip] {name} 없음");
            continue;
        }
        let mut b = match QwenBackend::new(&d, spec, Some("ko".into())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[skip] {name} 로드 실패: {e}");
                continue;
            }
        };
        let _ = b.transcribe_full(&audio[..SR.min(audio.len())], ""); // 워밍업
        let t0 = Instant::now();
        let mut text = String::new();
        let mut i = 0;
        while i < audio.len() {
            let end = (i + chunk).min(audio.len());
            if let Ok(s) = b.transcribe_full(&audio[i..end], "") {
                text.push_str(s.trim());
                text.push(' ');
            }
            i = end;
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        std::fs::write(format!("{OUT_DIR}/{name}.txt"), &text).ok();
        eprintln!(
            "[done] {name}: {:.1}s (RTF {:.2}, 30s청크 분할)",
            ms / 1000.0,
            ms / 1000.0 / audio_sec
        );
    }
}
