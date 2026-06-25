//! 전사 내보내기 포맷터 (txt/srt/json) — docs/02-architecture.md F·seed C11.

use asr_core::output::CommittedToken;

const SENTENCE_ENDS: [char; 6] = ['.', '?', '!', '。', '?', '!'];

fn speaker_label(s: Option<u32>) -> String {
    match s {
        Some(n) => format!("[화자 {}] ", n + 1),
        None => String::new(),
    }
}

pub fn to_txt(tokens: &[CommittedToken]) -> String {
    let mut out = String::new();
    let mut cur: Option<Option<u32>> = None;
    for t in tokens {
        if cur != Some(t.speaker) {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&speaker_label(t.speaker));
            cur = Some(t.speaker);
        }
        out.push_str(&t.text);
    }
    out.trim().to_string()
}

pub fn to_json(tokens: &[CommittedToken]) -> String {
    let text = to_txt(tokens);
    serde_json::json!({ "text": text, "tokens": tokens }).to_string()
}

/// 화자 변경 / 0.8s 공백 / 8토큰 / 문장부호 경계에서 cue 를 끊어 SRT 생성(화자 라벨 접두).
pub fn to_srt(tokens: &[CommittedToken]) -> String {
    let mut cues: Vec<(f64, f64, String)> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        let cur_start = i;
        let spk = tokens[i].speaker;
        let mut buf = String::new();
        let mut count = 0usize;
        loop {
            buf.push_str(&tokens[i].text);
            count += 1;
            let at_end = i + 1 == tokens.len();
            let ends_sentence = tokens[i].text.trim_end().ends_with(SENTENCE_ENDS);
            let gap_next = if at_end {
                f64::INFINITY
            } else {
                tokens[i + 1].start - tokens[i].end
            };
            let spk_change = !at_end && tokens[i + 1].speaker != spk;
            if at_end || ends_sentence || count >= 8 || gap_next > 0.8 || spk_change {
                break;
            }
            i += 1;
        }
        let text = buf.trim().to_string();
        if !text.is_empty() {
            cues.push((
                tokens[cur_start].start,
                tokens[i].end,
                format!("{}{}", speaker_label(spk), text),
            ));
        }
        i += 1;
    }

    let mut out = String::new();
    for (idx, (s, e, txt)) in cues.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            idx + 1,
            srt_time(*s),
            srt_time(*e),
            txt
        ));
    }
    out
}

fn srt_time(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as i64;
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1000;
    let mil = ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{mil:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: f64, e: f64, t: &str) -> CommittedToken {
        CommittedToken {
            start: s,
            end: e,
            text: t.to_string(),
            speaker: None,
        }
    }

    fn tok_spk(s: f64, e: f64, t: &str, spk: u32) -> CommittedToken {
        CommittedToken {
            start: s,
            end: e,
            text: t.to_string(),
            speaker: Some(spk),
        }
    }

    #[test]
    fn txt_labels_speakers() {
        let toks = vec![
            tok_spk(0.0, 0.5, "Hi", 0),
            tok_spk(0.6, 1.0, " there", 0),
            tok_spk(2.0, 2.4, "Hello", 1),
        ];
        let txt = to_txt(&toks);
        assert!(txt.starts_with("[화자 1] Hi there"), "{txt}");
        assert!(txt.contains("\n[화자 2] Hello"), "{txt}");
    }

    #[test]
    fn txt_joins_and_trims() {
        let toks = vec![tok(0.0, 0.5, " Hello"), tok(0.5, 1.0, " world.")];
        assert_eq!(to_txt(&toks), "Hello world.");
    }

    #[test]
    fn srt_splits_on_gap_and_punct() {
        let toks = vec![
            tok(0.0, 0.5, "Hello"),
            tok(0.5, 1.0, " world."),
            tok(3.0, 3.5, " Next"),
        ];
        let srt = to_srt(&toks);
        assert!(srt.contains("1\n00:00:00,000 --> 00:00:01,000\nHello world."), "{srt}");
        assert!(srt.contains("Next"), "{srt}");
    }

    #[test]
    fn json_has_text_and_tokens() {
        let toks = vec![tok(0.0, 0.5, "Hi")];
        let j = to_json(&toks);
        assert!(j.contains("\"text\""));
        assert!(j.contains("\"tokens\""));
        assert!(j.contains("Hi"));
    }

    #[test]
    fn srt_time_format() {
        assert_eq!(srt_time(3661.5), "01:01:01,500");
    }
}

