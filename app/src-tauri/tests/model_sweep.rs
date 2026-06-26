//! 모델 전수 벤치 — 회의 음성(meeting.wav)으로 모델별 레이턴시(RTF)·정확도(CER) 측정.
//! 정답 대본: 회의-01-제품팀-주간스프린트.md 의 **[화자]:** 대사.
//!
//! 실행: `cargo test -p app --test model_sweep -- --ignored --nocapture`
//! 결과 전사는 scratchpad 에 <model>.txt 로 저장(이후 정성 분석용).

use std::path::{Path, PathBuf};
use std::time::Instant;

use asr_core::asr::SelfStreamingBackend;
use asr_qwen::{QwenBackend, QWEN_06B};
use asr_sense::SenseVoiceBackend;
use asr_whisper::WhisperRsBackend;

const SR: usize = 16_000;
const OUT_DIR: &str = "/private/tmp/claude-501/-Users-kwonjunho-Desktop-work-tauri-stt-test/68e638b0-f4c7-4dbb-ba6f-f5c9ca7534ca/scratchpad/sweep";

/// 정답 대본에서 화자 대사만 추출해 하나의 텍스트로.
fn load_reference(md: &Path) -> String {
    let raw = std::fs::read_to_string(md).expect("md");
    let mut out = String::new();
    for line in raw.lines() {
        // 형식: **[박지민]:** 자, 다들 ...
        if let Some(idx) = line.find("]:**") {
            if line.trim_start().starts_with("**[") {
                out.push_str(&line[idx + 4..]);
                out.push(' ');
            }
        }
    }
    out
}

/// CER 비교용 정규화: 공백·문장부호 제거, 한글/숫자/영문(소문자)만 유지.
fn normalize(s: &str) -> Vec<char> {
    s.chars()
        .filter_map(|c| {
            if ('가'..='힣').contains(&c) || c.is_ascii_digit() {
                Some(c)
            } else if c.is_ascii_alphabetic() {
                Some(c.to_ascii_lowercase())
            } else {
                None
            }
        })
        .collect()
}

/// 레벤슈타인 편집거리(문자 단위).
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn cer(reference: &[char], hyp: &str) -> f64 {
    let h = normalize(hyp);
    if reference.is_empty() {
        return 1.0;
    }
    edit_distance(reference, &h) as f64 / reference.len() as f64
}

fn load_audio(wav: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(wav).expect("wav");
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

/// 한 모델 결과 한 줄.
struct Row {
    name: &'static str,
    params: &'static str,
    wall_ms: f64,
    rtf: f64,
    cer: f64,
    chars: usize,
}

#[test]
#[ignore = "회의 음성 + 모델 필요. --ignored 로 실행"]
fn model_sweep() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = base.join("..").join("..");
    let wav = base.join("test-data/meeting.wav");
    let md = root.join("회의-01-제품팀-주간스프린트.md");
    assert!(wav.exists(), "meeting.wav 없음");
    assert!(md.exists(), "대본 md 없음: {md:?}");
    std::fs::create_dir_all(OUT_DIR).ok();

    let audio = load_audio(&wav);
    let audio_sec = audio.len() as f64 / SR as f64;
    let reference = normalize(&load_reference(&md));
    eprintln!(
        "\n오디오 {:.0}s ({} 샘플), 정답 대본 {} 문자(정규화)\n",
        audio_sec,
        audio.len(),
        reference.len()
    );

    let mut rows: Vec<Row> = Vec::new();

    // ---- Whisper 계열(ggml, Metal): 풀패스 단일 전사 ----
    let whisper_models: &[(&str, &str, &str)] = &[
        ("ggml-tiny", "tiny 75M", "models/ggml/ggml-tiny.bin"),
        ("ggml-base", "base 141M", "models/ggml/ggml-base.bin"),
        ("ggml-small", "small 466M", "models/ggml/ggml-small.bin"),
        (
            "ggml-large-v3-turbo",
            "turbo 1.5G",
            "models/ggml/ggml-large-v3-turbo.bin",
        ),
        (
            "ggml-large-v3",
            "large-v3 3.1G",
            "models/ggml/ggml-large-v3.bin",
        ),
    ];
    for (name, params, rel) in whisper_models {
        let mp = base.join(rel);
        if !mp.exists() {
            eprintln!("[skip] {name} 파일 없음");
            continue;
        }
        let backend = match WhisperRsBackend::load(&mp, Some("ko".into())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[skip] {name} 로드 실패: {e}");
                continue;
            }
        };
        let _ = backend.transcribe(&audio[..SR.min(audio.len())], ""); // 워밍업(짧게)
        let t0 = Instant::now();
        let toks = backend.transcribe(&audio, "").expect("transcribe");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let text: String = toks.iter().map(|t| t.text.as_str()).collect();
        std::fs::write(format!("{OUT_DIR}/{name}.txt"), &text).ok();
        rows.push(Row {
            name,
            params,
            wall_ms: ms,
            rtf: ms / 1000.0 / audio_sec,
            cer: cer(&reference, &text),
            chars: normalize(&text).len(),
        });
        eprintln!("[done] {name}: {ms:.0}ms");
    }

    // ---- SenseVoice: 발화 모델 → 15s 청크 후 이어붙임 ----
    {
        let dir = base.join("models/sense");
        if dir.join("model.onnx").exists() {
            match SenseVoiceBackend::new(&dir, Some("ko".into())) {
                Ok(mut b) => {
                    let chunk = 15 * SR;
                    let t0 = Instant::now();
                    let mut text = String::new();
                    let mut i = 0;
                    while i < audio.len() {
                        let end = (i + chunk).min(audio.len());
                        if let Ok(s) = b.transcribe_full(&audio[i..end], "") {
                            text.push_str(&s);
                            text.push(' ');
                        }
                        i = end;
                    }
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    std::fs::write(format!("{OUT_DIR}/sensevoice.txt"), &text).ok();
                    rows.push(Row {
                        name: "sensevoice",
                        params: "SenseVoice 234M",
                        wall_ms: ms,
                        rtf: ms / 1000.0 / audio_sec,
                        cer: cer(&reference, &text),
                        chars: normalize(&text).len(),
                    });
                    eprintln!("[done] sensevoice: {ms:.0}ms");
                }
                Err(e) => eprintln!("[skip] sensevoice 로드 실패: {e}"),
            }
        }
    }

    // ---- Qwen3-ASR(0.6B): 풀패스(내부 세그먼트) ----
    let qwen_models: &[(&str, &str, &str, &asr_qwen::QwenModelSpec)] = &[
        ("qwen-0.6b", "Qwen3-ASR 0.6B", "models/qwen", &QWEN_06B),
    ];
    for (name, params, rel, spec) in qwen_models {
        let dir = base.join(rel);
        if !dir.join("config.json").exists() {
            eprintln!("[skip] {name} 없음");
            continue;
        }
        match QwenBackend::new(&dir, spec, Some("ko".into())) {
            Ok(mut b) => {
                let _ = b.transcribe_full(&audio[..SR.min(audio.len())], ""); // 워밍업
                let t0 = Instant::now();
                let text = b.transcribe_full(&audio, "").expect("qwen transcribe");
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                std::fs::write(format!("{OUT_DIR}/{name}.txt"), &text).ok();
                rows.push(Row {
                    name,
                    params,
                    wall_ms: ms,
                    rtf: ms / 1000.0 / audio_sec,
                    cer: cer(&reference, &text),
                    chars: normalize(&text).len(),
                });
                eprintln!("[done] {name}: {ms:.0}ms");
            }
            Err(e) => eprintln!("[skip] {name} 로드 실패: {e}"),
        }
    }

    // ---- 결과표 ----
    eprintln!("\n================ 모델 전수 벤치 (회의 {audio_sec:.0}s) ================");
    eprintln!(
        "{:<22} {:<18} {:>9} {:>7} {:>8} {:>8} {:>10}",
        "model", "params", "wall(s)", "RTF", "CER%", "정확%", "실시간?"
    );
    for r in &rows {
        // 실시간(스트리밍 12~15s 윈도우, 틱 1s): 틱당 추론 = window×RTF < 1s 필요 → RTF<~0.07
        let rt = if r.rtf < 0.07 {
            "✅ 여유"
        } else if r.rtf < 0.15 {
            "△ 빠듯"
        } else {
            "❌ 밀림"
        };
        eprintln!(
            "{:<22} {:<18} {:>9.1} {:>7.3} {:>7.1}% {:>7.1}% {:>10}",
            r.name,
            r.params,
            r.wall_ms / 1000.0,
            r.rtf,
            r.cer * 100.0,
            (1.0 - r.cer).max(0.0) * 100.0,
            rt
        );
    }
    eprintln!("==========================================================\n");
    eprintln!("전사 결과 저장: {OUT_DIR}/<model>.txt");
}
