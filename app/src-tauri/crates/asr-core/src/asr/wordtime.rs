//! 단어 분할 + 시간 가중 — WhisperLiveKit `qwen3_vllm_asr.py` 의 CJK 정렬 알고리즘 이식.
//!
//! 순수 알고리즘(OS 무관, 의존성 0). 두 가지 용도:
//! - `time_weight`: self_stream 의 토큰 타임스탬프를 **글자수 비례**로 배분(말한 길이 근사).
//!   기존의 단어-인덱스 균등 배분보다 경계가 실제에 가까워 화자분리 과분할을 줄인다.
//! - `split_align_words` / `fix_timestamps`: 향후 **실제 워드 얼라이너**(whisper 토큰
//!   타임스탬프, ForcedAligner 등)를 붙일 때 쓸 빌딩블록. 원본의 CJK 글자단위 분할과
//!   LIS(최장 비감소 부분수열) 기반 단조 보정을 그대로 보존한다.

/// CJK 통합 한자 영역인지(원본 `_is_cjk_char` 와 동일 범위). 한글(AC00–D7AF)은 제외 —
/// 한글 어절은 글자 단위로 쪼개지 않고 하나의 단어로 유지된다.
pub fn is_cjk_char(ch: char) -> bool {
    let c = ch as u32;
    (0x4E00..=0x9FFF).contains(&c)
        || (0x3400..=0x4DBF).contains(&c)
        || (0x20000..=0x2A6DF).contains(&c)
        || (0x2A700..=0x2B73F).contains(&c)
        || (0x2B740..=0x2B81F).contains(&c)
        || (0x2B820..=0x2CEAF).contains(&c)
        || (0xF900..=0xFAFF).contains(&c)
}

/// 정렬에 유지하는 문자(글자/숫자/어퍼스트로피). 원본 `_is_kept_char`:
/// 유니코드 카테고리 L*(글자)/N*(숫자) — Rust `is_alphanumeric` 과 대응.
pub fn is_kept_char(ch: char) -> bool {
    ch == '\'' || ch.is_alphanumeric()
}

/// 단어에서 유지문자만 남긴다(`_clean_align_token`).
pub fn clean_align_token(token: &str) -> String {
    token.chars().filter(|c| is_kept_char(*c)).collect()
}

/// 정렬용 단어 분할: 공백 분리 후 CJK 한자는 글자 단위, 그 외(한글·라틴 등)는 런 단위로 묶는다
/// (`_split_align_words`).
pub fn split_align_words(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    for segment in text.split_whitespace() {
        let cleaned = clean_align_token(segment);
        if cleaned.is_empty() {
            continue;
        }
        let mut buf = String::new();
        for ch in cleaned.chars() {
            if is_cjk_char(ch) {
                if !buf.is_empty() {
                    words.push(std::mem::take(&mut buf));
                }
                words.push(ch.to_string());
            } else {
                buf.push(ch);
            }
        }
        if !buf.is_empty() {
            words.push(buf);
        }
    }
    words
}

/// 시간 배분용 단어 가중치 = 유지문자(글자/숫자) 수, 최소 1. 한글 어절은 글자수≈음절수라
/// 말한 길이의 좋은 근사이며, 라틴 단어도 길이에 비례한다.
pub fn time_weight(word: &str) -> f64 {
    word.chars().filter(|c| is_kept_char(*c)).count().max(1) as f64
}

/// 비단조 타임스탬프를 LIS(최장 비감소 부분수열) 기준으로 보정+보간한다(`_fix_timestamps`).
/// LIS 에 속하지 않는(=역행/튐) 값만 좌우 정상값으로 보간 교체한다.
pub fn fix_timestamps(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut data: Vec<f64> = values.to_vec();
    if n <= 1 {
        return data;
    }
    let mut dp = vec![1usize; n];
    let mut parent = vec![-1i64; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j as i64;
            }
        }
    }
    // 최댓값의 첫 인덱스(Python list.index(max) 동작).
    let mut max_idx = 0usize;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }
    let mut lis: Vec<usize> = Vec::new();
    let mut cur = max_idx as i64;
    while cur != -1 {
        lis.push(cur as usize);
        cur = parent[cur as usize];
    }
    lis.reverse();
    let mut normal = vec![false; n];
    for &idx in &lis {
        normal[idx] = true;
    }

    let mut i = 0usize;
    while i < n {
        if normal[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !normal[j] {
            j += 1;
        }
        let count = j - i;
        let left = (0..i).rev().find(|&k| normal[k]).map(|k| data[k]);
        let right = (j..n).find(|&k| normal[k]).map(|k| data[k]);

        if count <= 2 {
            for k in i..j {
                match (left, right) {
                    (None, Some(r)) => data[k] = r,
                    (None, None) => {} // 양쪽 정상값 없음 → 그대로
                    (Some(l), None) => data[k] = l,
                    (Some(l), Some(r)) => {
                        let dl = k as i64 - (i as i64 - 1);
                        let dr = j as i64 - k as i64;
                        data[k] = if dl <= dr { l } else { r };
                    }
                }
            }
        } else if let (Some(l), Some(r)) = (left, right) {
            let step = (r - l) / (count as f64 + 1.0);
            for k in i..j {
                data[k] = l + step * ((k - i + 1) as f64);
            }
        } else if let Some(l) = left {
            for k in i..j {
                data[k] = l;
            }
        } else if let Some(r) = right {
            for k in i..j {
                data[k] = r;
            }
        }
        i = j;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_split_keeps_hangul_and_latin_as_runs() {
        // 한자는 글자단위, 한글 어절·라틴 단어는 하나로.
        assert_eq!(split_align_words("你好 world"), vec!["你", "好", "world"]);
        assert_eq!(split_align_words("안녕하세요 hi!"), vec!["안녕하세요", "hi"]);
    }

    #[test]
    fn time_weight_counts_kept_chars() {
        assert_eq!(time_weight("안녕하세요"), 5.0);
        assert_eq!(time_weight("hello"), 5.0);
        assert_eq!(time_weight("..."), 1.0); // 유지문자 0 → 최소 1
        assert_eq!(time_weight("don't"), 5.0); // 어퍼스트로피 유지
    }

    #[test]
    fn fix_timestamps_repairs_non_monotonic() {
        // [0,2,1,3]: LIS=[0,2,3](idx0,1,3), idx2(=1.0)만 보정 → 2.0
        let out = fix_timestamps(&[0.0, 2.0, 1.0, 3.0]);
        assert_eq!(out, vec![0.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn fix_timestamps_noop_when_monotonic() {
        let v = vec![0.0, 0.5, 1.0, 1.5];
        assert_eq!(fix_timestamps(&v), v);
    }
}
