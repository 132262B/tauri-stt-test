//! 경량 ASR 평가 지표 — 외부 의존 0.
//!
//! WhisperLiveKit `whisperlivekit/metrics.py` 이식: 단어 레벨 Levenshtein 으로 WER(치환/삽입/
//! 삭제 분해), 그리디 정렬 기반 단어 타임스탬프 정확도. 한국어용 문자 단위 CER 추가.
//!
//! 멀티플랫폼에서 가속 백엔드(Metal/Vulkan/CPU)별 전사 품질이 미세하게 달라질 수 있으므로
//! 타깃 간 정합성 회귀 베이스라인으로 쓴다(순수 알고리즘, OS 무관).
//!
//! 주: 원본의 unicode NFC 정규화는 의존성 회피를 위해 생략했다. whisper.cpp 출력은 통상
//! NFC 이므로 한·영 비교엔 영향이 적다. NFD 레퍼런스를 다루려면 normalize 단계에 NFC 추가.

/// WER 비교용 정규화: 소문자화 + 단어문자/공백/하이픈/어퍼스트로피만 남기고 공백 축약.
pub fn normalize_text(text: &str) -> String {
    let lowered = text.to_lowercase();
    let mut buf = String::with_capacity(lowered.len());
    for c in lowered.chars() {
        if c.is_alphanumeric() || c == '_' || c == '-' || c == '\'' {
            buf.push(c);
        } else {
            buf.push(' ');
        }
    }
    buf.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// WER(단어 오류율) 결과. wer 는 ref 단어수가 0이 아니면 dist/ref_words, 1.0 초과 가능.
#[derive(Debug, Clone, PartialEq)]
pub struct WerResult {
    pub wer: f64,
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub ref_words: usize,
    pub hyp_words: usize,
}

/// 단어 레벨 Levenshtein 편집거리로 WER 을 계산(치환/삽입/삭제 분해 포함).
pub fn compute_wer(reference: &str, hypothesis: &str) -> WerResult {
    let r = normalize_text(reference);
    let h = normalize_text(hypothesis);
    let ref_words: Vec<&str> = r.split_whitespace().collect();
    let hyp_words: Vec<&str> = h.split_whitespace().collect();
    let n = ref_words.len();
    let m = hyp_words.len();

    if n == 0 {
        return WerResult {
            wer: if m == 0 { 0.0 } else { m as f64 },
            substitutions: 0,
            insertions: m,
            deletions: 0,
            ref_words: 0,
            hyp_words: m,
        };
    }

    // dp[i][j] = (편집거리, 치환, 삽입, 삭제)
    let mut dp = vec![vec![(0usize, 0usize, 0usize, 0usize); m + 1]; n + 1];
    for i in 1..=n {
        dp[i][0] = (i, 0, 0, i);
    }
    for j in 1..=m {
        dp[0][j] = (j, 0, j, 0);
    }
    for i in 1..=n {
        for j in 1..=m {
            if ref_words[i - 1] == hyp_words[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                let sub = dp[i - 1][j - 1];
                let ins = dp[i][j - 1];
                let del = dp[i - 1][j];
                let sub_cost = (sub.0 + 1, sub.1 + 1, sub.2, sub.3);
                let ins_cost = (ins.0 + 1, ins.1, ins.2 + 1, ins.3);
                let del_cost = (del.0 + 1, del.1, del.2, del.3 + 1);
                // 동률(편집거리 같음)이면 원본과 동일하게 치환>삭제>삽입 순으로 선택.
                let mut best = sub_cost;
                if del_cost.0 < best.0 {
                    best = del_cost;
                }
                if ins_cost.0 < best.0 {
                    best = ins_cost;
                }
                dp[i][j] = best;
            }
        }
    }
    let (dist, subs, ins, dels) = dp[n][m];
    WerResult {
        wer: dist as f64 / n as f64,
        substitutions: subs,
        insertions: ins,
        deletions: dels,
        ref_words: n,
        hyp_words: m,
    }
}

/// CER(문자 오류율) 결과 — 한국어 등 띄어쓰기 모호한 언어용. 공백은 제거 후 비교.
#[derive(Debug, Clone, PartialEq)]
pub struct CerResult {
    pub cer: f64,
    pub edits: usize,
    pub ref_chars: usize,
    pub hyp_chars: usize,
}

/// 문자 단위 Levenshtein 으로 CER 을 계산(정규화 후 공백 제거).
pub fn compute_cer(reference: &str, hypothesis: &str) -> CerResult {
    let r: Vec<char> = normalize_text(reference).chars().filter(|c| !c.is_whitespace()).collect();
    let h: Vec<char> = normalize_text(hypothesis).chars().filter(|c| !c.is_whitespace()).collect();
    let n = r.len();
    let m = h.len();
    if n == 0 {
        return CerResult { cer: if m == 0 { 0.0 } else { m as f64 }, edits: m, ref_chars: 0, hyp_chars: m };
    }
    // 행 2개만 유지하는 표준 Levenshtein(메모리 O(m)).
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let edits = prev[m];
    CerResult { cer: edits as f64 / n as f64, edits, ref_chars: n, hyp_chars: m }
}

/// 타임스탬프 정확도 입력: 단어 + 시작/끝(초).
#[derive(Debug, Clone)]
pub struct TimedWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// 단어 타임스탬프 정확도(시작시각 델타 통계). 매칭이 없으면 None 들.
#[derive(Debug, Clone, PartialEq)]
pub struct TimestampAccuracy {
    pub mae_start: Option<f64>,
    pub max_delta_start: Option<f64>,
    pub median_delta_start: Option<f64>,
    pub n_matched: usize,
    pub n_ref: usize,
    pub n_pred: usize,
}

/// 예측 단어를 레퍼런스에 그리디(좌→우, 전방 3칸 탐색) 정렬해 시작시각 오차를 잰다.
pub fn compute_timestamp_accuracy(
    predicted: &[TimedWord],
    reference: &[TimedWord],
) -> TimestampAccuracy {
    let none = |np: usize, nr: usize| TimestampAccuracy {
        mae_start: None,
        max_delta_start: None,
        median_delta_start: None,
        n_matched: 0,
        n_ref: nr,
        n_pred: np,
    };
    if predicted.is_empty() || reference.is_empty() {
        return none(predicted.len(), reference.len());
    }

    let pred_norm: Vec<String> = predicted.iter().map(|p| normalize_text(&p.word)).collect();
    let ref_norm: Vec<String> = reference.iter().map(|r| normalize_text(&r.word)).collect();

    let mut deltas: Vec<f64> = Vec::new();
    let mut ref_idx = 0usize;
    for (p_idx, p_word) in pred_norm.iter().enumerate() {
        if p_word.is_empty() {
            continue;
        }
        let search_limit = (ref_idx + 3).min(ref_norm.len());
        for r_idx in ref_idx..search_limit {
            if &ref_norm[r_idx] == p_word {
                deltas.push(predicted[p_idx].start - reference[r_idx].start);
                ref_idx = r_idx + 1;
                break;
            }
        }
    }

    if deltas.is_empty() {
        return none(predicted.len(), reference.len());
    }

    let mut abs_deltas: Vec<f64> = deltas.iter().map(|d| d.abs()).collect();
    let sum: f64 = abs_deltas.iter().sum();
    let max = abs_deltas.iter().cloned().fold(f64::MIN, f64::max);
    abs_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = abs_deltas.len();
    let median = if k % 2 == 1 {
        abs_deltas[k / 2]
    } else {
        (abs_deltas[k / 2 - 1] + abs_deltas[k / 2]) / 2.0
    };

    TimestampAccuracy {
        mae_start: Some(sum / k as f64),
        max_delta_start: Some(max),
        median_delta_start: Some(median),
        n_matched: deltas.len(),
        n_ref: reference.len(),
        n_pred: predicted.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tw(word: &str, start: f64) -> TimedWord {
        TimedWord { word: word.into(), start, end: start + 0.3 }
    }

    #[test]
    fn normalize_strips_punct_and_case() {
        assert_eq!(normalize_text("Hello, World!"), "hello world");
        assert_eq!(normalize_text("  multiple   spaces  "), "multiple spaces");
    }

    #[test]
    fn wer_perfect_match_is_zero() {
        let r = compute_wer("the quick brown fox", "the quick brown fox");
        assert_eq!(r.wer, 0.0);
        assert_eq!(r.ref_words, 4);
        assert_eq!((r.substitutions, r.insertions, r.deletions), (0, 0, 0));
    }

    #[test]
    fn wer_counts_sub_ins_del() {
        // ref: a b c d   hyp: a x c d e  -> 1 substitution(b->x) + 1 insertion(e)
        let r = compute_wer("a b c d", "a x c d e");
        assert_eq!(r.substitutions, 1);
        assert_eq!(r.insertions, 1);
        assert_eq!(r.deletions, 0);
        assert_eq!(r.ref_words, 4);
        assert!((r.wer - 0.5).abs() < 1e-9);
    }

    #[test]
    fn wer_empty_reference() {
        let r = compute_wer("", "two words");
        assert_eq!(r.insertions, 2);
        assert_eq!(r.wer, 2.0);
    }

    #[test]
    fn cer_korean_chars() {
        // 1 글자 치환: 안녕하세요 -> 안뇽하세요
        let r = compute_cer("안녕하세요", "안뇽하세요");
        assert_eq!(r.ref_chars, 5);
        assert_eq!(r.edits, 1);
        assert!((r.cer - 0.2).abs() < 1e-9);
    }

    #[test]
    fn timestamp_accuracy_greedy() {
        let reference = vec![tw("the", 0.0), tw("cat", 1.0), tw("sat", 2.0)];
        let predicted = vec![tw("the", 0.1), tw("cat", 1.2), tw("sat", 1.9)];
        let a = compute_timestamp_accuracy(&predicted, &reference);
        assert_eq!(a.n_matched, 3);
        // 델타 abs: 0.1, 0.2, 0.1 → mae≈0.133, median=0.1, max=0.2
        assert!((a.mae_start.unwrap() - 0.4 / 3.0).abs() < 1e-9);
        assert!((a.median_delta_start.unwrap() - 0.1).abs() < 1e-9);
        assert!((a.max_delta_start.unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn timestamp_accuracy_no_match() {
        let reference = vec![tw("alpha", 0.0)];
        let predicted = vec![tw("zzz", 0.0)];
        let a = compute_timestamp_accuracy(&predicted, &reference);
        assert_eq!(a.n_matched, 0);
        assert!(a.mae_start.is_none());
    }
}
