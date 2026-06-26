//! Rust 네이티브 Whisper ASR — whisper.cpp(whisper-rs, Metal)로 in-process 전사.
//!
//! Python 사이드카를 대체한다. SelfStreaming(전체 텍스트 공통접두사 확정) 으로 감싸
//! 한국어 어순 뒤섞임 없이 스트리밍 확정/partial 을 만든다.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use asr_core::asr::{
    AsrConfig, AsrError, AsrProfile, AsrToken, BackendCaps, SelfStreamingBackend,
    SelfStreamingProcessor, StreamingAsrBackend,
};
use async_trait::async_trait;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// 단어/토큰 단위 전사 결과(절대시각 초).
#[derive(Clone, Debug)]
pub struct WhisperToken {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct WhisperSegmentText {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Clone, Copy, Debug)]
pub struct WhisperDecodeOptions {
    pub n_threads: i32,
    pub token_timestamps: bool,
    pub no_timestamps: bool,
    pub single_segment: bool,
    pub no_context: bool,
    pub audio_ctx: Option<i32>,
    pub max_tokens: Option<i32>,
    pub n_max_text_ctx: Option<i32>,
    pub temperature: f32,
    pub temperature_inc: f32,
    pub require_coreml_encoder: bool,
    pub use_gpu: bool,
}

impl Default for WhisperDecodeOptions {
    fn default() -> Self {
        Self {
            n_threads: default_threads(),
            token_timestamps: true,
            no_timestamps: false,
            single_segment: false,
            no_context: false,
            audio_ctx: None,
            max_tokens: None,
            n_max_text_ctx: None,
            temperature: 0.0,
            temperature_inc: 0.2,
            require_coreml_encoder: false,
            use_gpu: whisper_use_gpu(),
        }
    }
}

impl WhisperDecodeOptions {
    pub fn for_profile(profile: AsrProfile) -> Self {
        match profile {
            AsrProfile::RealtimeQ5 => {
                let use_coreml = q5_use_coreml_encoder();
                Self {
                    token_timestamps: false,
                    no_timestamps: true,
                    single_segment: true,
                    no_context: true,
                    audio_ctx: Some(if use_coreml { 1024 } else { 512 }),
                    max_tokens: Some(64),
                    n_max_text_ctx: Some(0),
                    temperature_inc: 0.0,
                    require_coreml_encoder: use_coreml && !allow_q5_without_coreml(),
                    use_gpu: realtime_q5_use_gpu(),
                    ..Self::default()
                }
            }
            AsrProfile::Auto | AsrProfile::Balanced => Self::default(),
        }
    }

    pub fn for_windowed_q5() -> Self {
        let use_coreml = q5_use_coreml_encoder();
        Self {
            token_timestamps: false,
            no_timestamps: true,
            single_segment: true,
            no_context: true,
            audio_ctx: Some(if use_coreml {
                1024
            } else {
                q5_windowed_audio_ctx()
            }),
            max_tokens: Some(96),
            n_max_text_ctx: Some(0),
            temperature_inc: q5_temp_inc(),
            require_coreml_encoder: use_coreml && !allow_q5_without_coreml(),
            use_gpu: realtime_q5_use_gpu(),
            ..Self::default()
        }
    }
}

#[cfg(target_os = "ios")]
fn default_threads() -> i32 {
    3
}

#[cfg(not(target_os = "ios"))]
fn default_threads() -> i32 {
    4
}

/// whisper.cpp 모델을 적재해 전사를 수행하는 백엔드.
pub struct WhisperRsBackend {
    ctx: WhisperContext,
    language: Option<String>,
    /// whisper_state 재사용은 CoreML 경로에서 이전 full() 결과를 그대로 반환하는 케이스가
    /// 있어 실제 스트리밍에서 첫 문장 반복을 만든다. context/model만 유지하고 state는
    /// 호출마다 새로 만든다.
    state_lock: std::sync::Mutex<()>,
}

impl WhisperRsBackend {
    pub fn load(model_path: impl AsRef<Path>, language: Option<String>) -> Result<Self, String> {
        Self::load_with_options(model_path, language, WhisperDecodeOptions::default())
    }

    pub fn load_with_options(
        model_path: impl AsRef<Path>,
        language: Option<String>,
        options: WhisperDecodeOptions,
    ) -> Result<Self, String> {
        ensure_whisper_logging_hooks();
        let mut model_path = model_path.as_ref().to_path_buf();
        let mut options = options;
        if is_q5_turbo_model_path(&model_path) && !q5_use_coreml_encoder() {
            match no_coreml_model_alias(&model_path) {
                Ok(alias) => {
                    eprintln!(
                        "[whisper] Q5 CoreML encoder disabled by default; using CPU encoder alias: {}",
                        alias.display()
                    );
                    model_path = alias;
                }
                Err(e) => {
                    eprintln!("[whisper] Q5 CPU alias 생성 실패, CoreML path fallback 사용: {e}");
                    if coreml_encoder_path(&model_path).exists() {
                        options.audio_ctx = Some(1024);
                    }
                }
            }
        }
        let coreml = coreml_encoder_path(&model_path);
        if options.require_coreml_encoder {
            if !coreml.exists() {
                return Err(format!(
                    "Q5 realtime 프로파일은 CoreML encoder가 필요합니다: {} (임시 CPU/Metal fallback은 ASR_Q5_ALLOW_NO_COREML=1)",
                    coreml.display()
                ));
            }
        }
        if coreml.exists() {
            eprintln!(
                "[whisper] CoreML encoder candidate: {} | use_gpu={} ({})",
                coreml.display(),
                options.use_gpu,
                whisper_rs::print_system_info()
            );
        }
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(options.use_gpu);
        if options.use_gpu && whisper_flash_attn() {
            ctx_params.flash_attn(true);
        }
        let ctx =
            WhisperContext::new_with_params(model_path.to_string_lossy().as_ref(), ctx_params)
                .map_err(|e| format!("whisper 모델 적재 실패: {e}"))?;
        Ok(Self {
            ctx,
            language,
            state_lock: std::sync::Mutex::new(()),
        })
    }

    pub fn set_language(&mut self, lang: Option<String>) {
        self.language = lang;
    }

    pub fn warmup(&self) -> Result<(), String> {
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "state 잠금 실패".to_string())?;
        let _state = self
            .ctx
            .create_state()
            .map_err(|e| format!("state 생성 실패: {e}"))?;
        Ok(())
    }

    /// 16kHz mono f32 PCM 전체를 전사 → 토큰 리스트(특수토큰 제외).
    pub fn transcribe(
        &self,
        audio: &[f32],
        init_prompt: &str,
    ) -> Result<Vec<WhisperToken>, String> {
        self.transcribe_with_options(audio, init_prompt, WhisperDecodeOptions::default())
    }

    /// 16kHz mono f32 PCM 전체를 전사 → 토큰 리스트(특수토큰 제외).
    pub fn transcribe_with_options(
        &self,
        audio: &[f32],
        init_prompt: &str,
        options: WhisperDecodeOptions,
    ) -> Result<Vec<WhisperToken>, String> {
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "state 잠금 실패".to_string())?;
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("state 생성 실패: {e}"))?;

        let params = self.params(init_prompt, options);

        state
            .full(params, audio)
            .map_err(|e| format!("전사 실패: {e}"))?;

        // whisper-rs 0.16 객체형 API: get_segment → WhisperSegment → get_token.
        let n_segments = state.full_n_segments();
        let mut tokens = Vec::new();
        for s in 0..n_segments {
            let Some(seg) = state.get_segment(s) else {
                continue;
            };
            // 무음/비음성 세그먼트 스킵(무음 구간 가짜 화자 라벨도 함께 제거)
            if seg.no_speech_probability() > 0.6 {
                continue;
            }
            if let Ok(st) = seg.to_str() {
                let st = st.trim();
                if st.is_empty() || st.starts_with('[') || st.starts_with('(') {
                    continue;
                }
            }
            let n_tok = seg.n_tokens();
            for ti in 0..n_tok {
                let Some(tok) = seg.get_token(ti) else {
                    continue;
                };
                let text = match tok.to_str() {
                    Ok(s) => s.to_string(),
                    Err(_) => continue,
                };
                // 특수/주석 토큰 제외([_BEG_], <|..|>, [BLANK_AUDIO] 조각, (..))
                let tt = text.trim_start();
                if tt.starts_with("[_")
                    || tt.starts_with("<|")
                    || tt.starts_with('[')
                    || tt.starts_with('(')
                {
                    continue;
                }
                let data = tok.token_data(); // t0/t1: centisecond(10ms)
                let start = data.t0 as f64 / 100.0;
                let end = data.t1 as f64 / 100.0;
                tokens.push(WhisperToken { start, end, text });
            }
        }
        Ok(tokens)
    }

    /// SelfStreaming 경로용 빠른 텍스트 전사. 토큰 타임스탬프 추출을 건너뛸 수 있다.
    pub fn transcribe_text(
        &self,
        audio: &[f32],
        init_prompt: &str,
        options: WhisperDecodeOptions,
    ) -> Result<String, String> {
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "state 잠금 실패".to_string())?;
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("state 생성 실패: {e}"))?;
        let params = self.params(init_prompt, options);

        state
            .full(params, audio)
            .map_err(|e| format!("전사 실패: {e}"))?;

        let n_segments = state.full_n_segments();
        let mut text = String::new();
        for s in 0..n_segments {
            let Some(seg) = state.get_segment(s) else {
                continue;
            };
            if seg.no_speech_probability() > 0.6 {
                continue;
            }
            let Ok(seg_text) = seg.to_str() else { continue };
            let st = seg_text.trim();
            if st.is_empty() || st.starts_with('[') || st.starts_with('(') {
                continue;
            }
            text.push_str(seg_text);
        }
        Ok(clean_transcript_text(&text))
    }

    pub fn transcribe_segments(
        &self,
        audio: &[f32],
        init_prompt: &str,
        options: WhisperDecodeOptions,
    ) -> Result<Vec<WhisperSegmentText>, String> {
        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| "state 잠금 실패".to_string())?;
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("state 생성 실패: {e}"))?;
        let params = self.params(init_prompt, options);

        state
            .full(params, audio)
            .map_err(|e| format!("전사 실패: {e}"))?;

        let n_segments = state.full_n_segments();
        let mut segments = Vec::new();
        for s in 0..n_segments {
            let Some(seg) = state.get_segment(s) else {
                continue;
            };
            if seg.no_speech_probability() > 0.6 {
                continue;
            }
            let Ok(seg_text) = seg.to_str() else { continue };
            let text = clean_transcript_text(seg_text);
            let text = text.trim();
            if text.is_empty() || text.starts_with('[') || text.starts_with('(') {
                continue;
            }
            let start = (seg.start_timestamp().max(0) as f64) / 100.0;
            let end = (seg.end_timestamp().max(seg.start_timestamp()) as f64) / 100.0;
            segments.push(WhisperSegmentText {
                start,
                end,
                text: text.to_string(),
            });
        }
        Ok(segments)
    }

    fn params<'a>(
        &'a self,
        init_prompt: &'a str,
        options: WhisperDecodeOptions,
    ) -> FullParams<'a, 'a> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(options.n_threads);
        params.set_translate(false);
        // whisper.cpp 기본 언어는 "en" 이라, None(자동) 일 때 반드시 "auto" 를 명시해야
        // 언어 자동 감지가 동작한다. (이 설정이 없으면 한국어 발화가 영어로 전사됨)
        match &self.language {
            Some(l) => params.set_language(Some(l.as_str())),
            None => params.set_language(Some("auto")),
        }
        if !init_prompt.is_empty() {
            params.set_initial_prompt(init_prompt);
        }
        params.set_token_timestamps(options.token_timestamps);
        params.set_no_timestamps(options.no_timestamps);
        params.set_single_segment(options.single_segment);
        params.set_no_context(options.no_context);
        params.set_temperature(options.temperature);
        params.set_temperature_inc(options.temperature_inc);
        if let Some(audio_ctx) = options.audio_ctx {
            params.set_audio_ctx(audio_ctx);
        }
        if let Some(max_tokens) = options.max_tokens {
            params.set_max_tokens(max_tokens);
        }
        if let Some(n_max_text_ctx) = options.n_max_text_ctx {
            params.set_n_max_text_ctx(n_max_text_ctx);
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params
    }
}

fn coreml_encoder_path(model_path: &Path) -> PathBuf {
    let Some(parent) = model_path.parent() else {
        return PathBuf::from("ggml-encoder.mlmodelc");
    };
    let stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("ggml");
    let base = strip_ggml_quant_suffix(stem);
    parent.join(format!("{base}-encoder.mlmodelc"))
}

fn strip_ggml_quant_suffix(stem: &str) -> &str {
    let bytes = stem.as_bytes();
    if bytes.len() >= 5 {
        let start = bytes.len() - 5;
        if bytes[start] == b'-'
            && bytes[start + 1] == b'q'
            && bytes[start + 2].is_ascii_digit()
            && bytes[start + 3] == b'_'
            && bytes[start + 4].is_ascii_digit()
        {
            return &stem[..start];
        }
    }
    stem
}

fn allow_q5_without_coreml() -> bool {
    env_bool("ASR_Q5_ALLOW_NO_COREML").unwrap_or(false)
}

fn q5_use_coreml_encoder() -> bool {
    env_bool("ASR_Q5_USE_COREML").unwrap_or(false)
}

fn is_q5_turbo_model_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            let name = name.to_ascii_lowercase();
            name.contains("large-v3-turbo") && name.contains("q5_0")
        })
        .unwrap_or(false)
}

fn no_coreml_model_alias(model_path: &Path) -> Result<PathBuf, String> {
    let file_name = model_path
        .file_name()
        .ok_or_else(|| format!("모델 파일명 확인 실패: {}", model_path.display()))?;
    let alias_dir = std::env::temp_dir().join("tauri-stt-q5-no-coreml");
    std::fs::create_dir_all(&alias_dir)
        .map_err(|e| format!("{} 생성 실패: {e}", alias_dir.display()))?;
    let alias = alias_dir.join(file_name);
    if let Ok(target) = std::fs::read_link(&alias) {
        if target == model_path {
            return Ok(alias);
        }
        let _ = std::fs::remove_file(&alias);
    } else if alias.exists() {
        return Ok(alias);
    }

    create_model_alias(model_path, &alias)?;
    Ok(alias)
}

#[cfg(unix)]
fn create_model_alias(model_path: &Path, alias: &Path) -> Result<(), String> {
    std::os::unix::fs::symlink(model_path, alias).map_err(|e| {
        format!(
            "{} -> {} symlink 실패: {e}",
            alias.display(),
            model_path.display()
        )
    })
}

#[cfg(not(unix))]
fn create_model_alias(model_path: &Path, alias: &Path) -> Result<(), String> {
    std::fs::hard_link(model_path, alias).map_err(|e| {
        format!(
            "{} -> {} hard link 실패: {e}",
            alias.display(),
            model_path.display()
        )
    })
}

fn clean_transcript_text(text: &str) -> String {
    let text = text.trim();
    if likely_repetitive_hallucination(text) {
        return String::new();
    }
    collapse_immediate_repeats(text)
}

/// 같은 단어가 연속 4회 이상 반복되면(환각 stutter: "개처럼 개처럼 개처럼 개처럼…") 1회로 접는다.
/// 정상 발화의 강조 중복(2~3회)은 보존. 구두점은 무시하고 정규화 비교. 전사 결과를 막지 않고
/// 보이는 반복만 정리하는 보수적 안전망(근본 해결은 temperature 폴백·audio_ctx).
fn collapse_immediate_repeats(text: &str) -> String {
    const RUN: usize = 4;
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < RUN {
        return text.to_string();
    }
    let mut out: Vec<&str> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let norm = normalize_word_for_overlap(words[i]);
        let mut j = i + 1;
        while j < words.len()
            && !norm.is_empty()
            && normalize_word_for_overlap(words[j]) == norm
        {
            j += 1;
        }
        if j - i >= RUN {
            out.push(words[i]); // 연속 반복 → 1회로 접음
        } else {
            out.extend_from_slice(&words[i..j]);
        }
        i = j;
    }
    out.join(" ")
}

fn likely_repetitive_hallucination(text: &str) -> bool {
    let words = normalized_words(text);
    if !words.is_empty() && words.len() <= 4 && words.iter().all(|word| is_short_filler_word(word))
    {
        return true;
    }
    if words.len() >= 8 {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for word in &words {
            *counts.entry(word.clone()).or_default() += 1;
        }
        let max_count = counts.values().copied().max().unwrap_or(0);
        if max_count >= 8 && max_count * 100 >= words.len() * 70 {
            return true;
        }
        if words.len() >= 16 && counts.len() <= 3 {
            return true;
        }
    }

    let mut char_counts: HashMap<char, usize> = HashMap::new();
    let mut total = 0usize;
    for ch in text
        .chars()
        .filter(|ch| !ch.is_whitespace() && !is_noise_punct(*ch))
    {
        total += 1;
        *char_counts.entry(ch).or_default() += 1;
    }
    total >= 24 && char_counts.values().copied().max().unwrap_or(0) * 100 >= total * 75
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|word| {
            let word = word
                .trim_matches(is_noise_punct)
                .trim()
                .to_ascii_lowercase();
            if word.is_empty() {
                None
            } else {
                Some(word)
            }
        })
        .collect()
}

fn is_noise_punct(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.'
            | '，'
            | '。'
            | '、'
            | '!'
            | '?'
            | '！'
            | '？'
            | '…'
            | ':'
            | ';'
            | '"'
            | '\''
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '<'
            | '>'
            | '·'
            | '-'
            | '_'
            | '~'
    )
}

fn is_short_filler_word(word: &str) -> bool {
    matches!(
        word,
        "아" | "어" | "음" | "으" | "흠" | "ah" | "uh" | "um" | "hmm"
    )
}

static WHISPER_LOGGING_HOOKS: Once = Once::new();

fn ensure_whisper_logging_hooks() {
    if !env_bool("ASR_WHISPER_VERBOSE").unwrap_or(false) {
        WHISPER_LOGGING_HOOKS.call_once(whisper_rs::install_logging_hooks);
    }
}

fn whisper_use_gpu() -> bool {
    env_bool("ASR_WHISPER_USE_GPU").unwrap_or(false)
}

fn realtime_q5_use_gpu() -> bool {
    // 라이브 Q5 인코더/디코더를 Metal GPU로 실행한다. CPU 단독이면 6초 윈도우 디코드가
    // STEP(2s)을 넘겨 지연이 누적된다(벤치 90/90 실시간 미추종). whisper-rs는
    // features=["metal"]로 이미 컴파일돼 있어 재빌드 없이 런타임 플래그만 켜면 된다.
    // 데스크톱(Apple Silicon = 라이브 Q5 타깃)은 기본 ON, 검증 전인 iOS는 env opt-in.
    // 문제 시 ASR_WHISPER_USE_GPU=0 으로 즉시 되돌릴 수 있다.
    env_bool("ASR_WHISPER_USE_GPU").unwrap_or(cfg!(not(target_os = "ios")))
}

/// Metal flash attention(융합 어텐션). GPU일 때만 의미가 있고 DTW 미사용이라 호환.
/// 기본 OFF — 같은 GPU 빌드에서 on/off A/B 측정용 opt-in.
fn whisper_flash_attn() -> bool {
    env_bool("ASR_WHISPER_FLASH_ATTN").unwrap_or(false)
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|v| match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok())
}

/// 윈도우(<=6s) Q5 인코더 audio_ctx. 기본 512. 더 줄이면(예: 320) 인코더는 빨라지지만
/// 모델이 학습한 컨텍스트(1500)보다 과도하게 짧아져 한국어 반복/환각 루프가 늘어난다
/// (실측: 320에서 "괜찮은데? 괜찮은데?…" 류 반복 폭증) — 속도가 더 필요할 때만 env 로 낮출 것.
/// 6초=~300 인코더프레임이므로 300 미만이면 실오디오가 잘린다. env Q5_AUDIO_CTX.
fn q5_windowed_audio_ctx() -> i32 {
    std::env::var("Q5_AUDIO_CTX")
        .ok()
        .and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(512)
}

/// 윈도우 Q5 디코드 temperature 증가폭(폴백). >0 이면 whisper 가 반복/저신뢰 세그먼트를
/// compression-ratio/entropy 로 감지해 더 높은 temperature 로 재디코드 → 반복 환각 루프를
/// 탈출한다(현실 회의/캐주얼 음성에서 중요). 0 이면 폴백 없음(가장 빠르나 루프에 빠지면
/// 못 빠져나옴). Metal 가속으로 재디코드 비용을 감당할 수 있으므로 기본 0.2. env Q5_TEMP_INC.
fn q5_temp_inc() -> f32 {
    std::env::var("Q5_TEMP_INC")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.2)
}

fn normalized_language(lang: Option<String>) -> Option<String> {
    lang.and_then(|s| {
        let s = s.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            None
        } else {
            Some(s.to_string())
        }
    })
}

fn language_for_profile(profile: AsrProfile, lang: Option<String>) -> Option<String> {
    if profile == AsrProfile::RealtimeQ5 {
        None
    } else {
        normalized_language(lang)
    }
}

/// ggml 모델을 dir 에서 찾고, 없으면 HF(ggerganov/whisper.cpp)에서 다운로드.
fn resolve_ggml(dir: &Path, model_id: &str) -> Result<PathBuf, AsrError> {
    std::fs::create_dir_all(dir).ok();
    let name = if model_id.ends_with(".bin") {
        model_id.to_string()
    } else {
        format!("{model_id}.bin")
    };
    let path = dir.join(&name);
    if path.exists() {
        return Ok(path);
    }
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}");
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| AsrError::ModelMissing(format!("{name} 다운로드 실패: {e}")))?;
    // 기대 크기(Content-Length): 다운로드가 잘리면 깨진 모델 → whisper.cpp 가
    // GGML_ASSERT 로 프로세스를 abort 시키므로, 로드 전에 크기로 무결성을 검증한다.
    let expected: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let tmp = path.with_extension("part");
    let mut f = std::fs::File::create(&tmp).map_err(|e| AsrError::Inference(e.to_string()))?;
    let written = std::io::copy(&mut resp.into_reader(), &mut f)
        .map_err(|e| AsrError::Inference(e.to_string()))?;
    drop(f);
    if let Some(exp) = expected {
        if written != exp {
            let _ = std::fs::remove_file(&tmp);
            return Err(AsrError::ModelMissing(format!(
                "{name} 다운로드 손상: {written}B 받음(기대 {exp}B). 다시 시도하세요."
            )));
        }
    }
    std::fs::rename(&tmp, &path).map_err(|e| AsrError::Inference(e.to_string()))?;
    Ok(path)
}

/// Whisper 를 발화 단위(전체 텍스트) 백엔드로 노출 — SelfStreamingProcessor 가 연속 2회
/// 전사의 공통 단어 접두사만 확정한다. 토큰 타임스탬프에 의존하지 않으므로 한국어에서
/// 조각/중복/어순 뒤섞임이 없다(LocalAgreement 의 토큰-타임스탬프 의존 문제 회피).
pub struct WhisperSelfBackend {
    models_dir: PathBuf,
    language: Option<String>,
    force_auto_language: bool,
    backend: Option<WhisperRsBackend>,
    decode_options: WhisperDecodeOptions,
}

impl WhisperSelfBackend {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            language: None,
            force_auto_language: false,
            backend: None,
            decode_options: WhisperDecodeOptions::default(),
        }
    }
}

impl SelfStreamingBackend for WhisperSelfBackend {
    fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        let path = resolve_ggml(&self.models_dir, &cfg.model_id)?;
        let profile = cfg.effective_profile();
        self.force_auto_language = profile == AsrProfile::RealtimeQ5;
        self.language = language_for_profile(profile, cfg.language.clone());
        self.decode_options = WhisperDecodeOptions::for_profile(profile);
        self.backend = Some(
            WhisperRsBackend::load_with_options(&path, self.language.clone(), self.decode_options)
                .map_err(AsrError::Inference)?,
        );
        Ok(())
    }

    fn transcribe_full(&mut self, samples: &[f32], prompt: &str) -> Result<String, AsrError> {
        let b = self.backend.as_ref().ok_or(AsrError::NotReady)?;
        b.transcribe_text(samples, prompt, self.decode_options)
            .map_err(AsrError::Inference)
    }

    fn warmup(&mut self) -> Result<(), AsrError> {
        let b = self.backend.as_ref().ok_or(AsrError::NotReady)?;
        b.warmup().map_err(AsrError::Inference)
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.language = if self.force_auto_language {
            None
        } else {
            normalized_language(lang)
        };
        if let Some(b) = &mut self.backend {
            b.set_language(self.language.clone());
        }
    }
}

struct WhisperAdaptiveBackend {
    models_dir: PathBuf,
    inner: Option<Box<dyn StreamingAsrBackend>>,
}

impl WhisperAdaptiveBackend {
    fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            inner: None,
        }
    }

    fn inner_mut(&mut self) -> Result<&mut Box<dyn StreamingAsrBackend>, AsrError> {
        self.inner.as_mut().ok_or(AsrError::NotReady)
    }
}

#[async_trait]
impl StreamingAsrBackend for WhisperAdaptiveBackend {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        let mut inner: Box<dyn StreamingAsrBackend> =
            if cfg.effective_profile() == AsrProfile::RealtimeQ5 {
                Box::new(WhisperWindowedBackend::new(self.models_dir.clone()))
            } else {
                Box::new(SelfStreamingProcessor::new(WhisperSelfBackend::new(
                    self.models_dir.clone(),
                )))
            };
        inner.configure(cfg).await?;
        self.inner = Some(inner);
        Ok(())
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        self.inner_mut()?.warmup().await
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64) {
        if let Some(inner) = &mut self.inner {
            inner.insert_audio_chunk(pcm, end_time);
        }
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        self.inner_mut()?.process_iter(is_last).await
    }

    fn get_buffer(&self) -> String {
        self.inner
            .as_ref()
            .map(|inner| inner.get_buffer())
            .unwrap_or_default()
    }

    fn set_language(&mut self, lang: Option<String>) {
        if let Some(inner) = &mut self.inner {
            inner.set_language(lang);
        }
    }

    fn caps(&self) -> BackendCaps {
        self.inner
            .as_ref()
            .map(|inner| inner.caps())
            .unwrap_or(BackendCaps {
                provides_word_timestamps: false,
                provides_probability: false,
                self_streaming: true,
                tokenizer_id: "whisper_adaptive",
            })
    }
}

// MIN/STEP/MAX/HOLDBACK/BACKLOG/MAX_LAG 은 WhisperWindowedBackend 의 기본값으로,
// 각각 Q5_WIN_MIN_SEC/STEP_SEC/MAX_SEC/HOLDBACK/BACKLOG_SEC/MAX_LAG_SEC env 로 재빌드 없이 덮어쓸 수 있다.
const WINDOWED_Q5_MIN_SEC: f64 = 5.0;
const WINDOWED_Q5_STEP_SEC: f64 = 2.0;
const WINDOWED_Q5_MAX_SEC: f64 = 6.0;
const WINDOWED_Q5_BACKLOG_SEC: f64 = 120.0;
const WINDOWED_Q5_HOLDBACK_WORDS: usize = 1;
/// 라이브 디코드가 이만큼(초) 이상 실시간보다 뒤처지면 최신 윈도우로 점프해 따라잡는다
/// (중간 구간 누락). 느린 디코드에도 라이브가 '지금'을 따라가고, 정지 시 밀린 백로그를
/// 길게 토해내지 않게 하는 상한. 0 이면 비활성.
const WINDOWED_Q5_MAX_LAG_SEC: f64 = 6.0;
const WINDOWED_Q5_DUP_MIN_WORDS: usize = 4;
const WINDOWED_Q5_GAP_DIAG_SEC: f64 = 1.25;
const WINDOWED_Q5_ACTIVE_RMS: f32 = 0.0015;

#[derive(Clone, Debug)]
struct WindowWord {
    text: String,
    norm: String,
}

struct WhisperWindowedBackend {
    models_dir: PathBuf,
    backend: Option<WhisperRsBackend>,
    decode_options: WhisperDecodeOptions,
    audio: Vec<f32>,
    audio_offset: f64,
    last_decode_end: f64,
    buffer: String,
    last_committed_end: f64,
    committed_norm_words: Vec<String>,
    // 라이브 지연 튜닝 노브(기본값=위 상수). 재빌드 없이 env로 스윕 가능:
    //   Q5_WIN_MIN_SEC   첫 디코드까지 누적해야 할 오디오(첫 토큰 바닥). ↓이면 더 빨리 뜨나 짧은 첫 윈도우 환각 위험.
    //   Q5_WIN_STEP_SEC  확정 전진 간격(케이던스). ↓이면 더 자주 확정/partial 갱신, 단 디코드 부하↑(I<STEP 유지 필요).
    //   Q5_WIN_MAX_SEC   디코드 윈도우 길이(문맥 span). 보통 6 유지.
    //   Q5_WIN_HOLDBACK  윈도우 끝에서 보류할 단어 수(가장자리 흔들림 가림). 0이면 확정 지연↓·깜빡임 위험.
    //   Q5_WIN_MAX_LAG_SEC  라이브가 이만큼 뒤처지면 최신으로 점프(중간 누락). 정지 즉시성·라이브성 보장.
    min_sec: f64,
    step_sec: f64,
    max_sec: f64,
    holdback_words: usize,
    backlog_sec: f64,
    max_lag_sec: f64,
}

impl WhisperWindowedBackend {
    fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            backend: None,
            decode_options: WhisperDecodeOptions::for_windowed_q5(),
            audio: Vec::new(),
            audio_offset: 0.0,
            last_decode_end: 0.0,
            buffer: String::new(),
            last_committed_end: 0.0,
            committed_norm_words: Vec::new(),
            min_sec: env_f64("Q5_WIN_MIN_SEC").unwrap_or(WINDOWED_Q5_MIN_SEC),
            step_sec: env_f64("Q5_WIN_STEP_SEC").unwrap_or(WINDOWED_Q5_STEP_SEC),
            max_sec: env_f64("Q5_WIN_MAX_SEC").unwrap_or(WINDOWED_Q5_MAX_SEC),
            holdback_words: env_usize("Q5_WIN_HOLDBACK").unwrap_or(WINDOWED_Q5_HOLDBACK_WORDS),
            backlog_sec: env_f64("Q5_WIN_BACKLOG_SEC").unwrap_or(WINDOWED_Q5_BACKLOG_SEC),
            max_lag_sec: std::env::var("Q5_WIN_MAX_LAG_SEC")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(WINDOWED_Q5_MAX_LAG_SEC),
        }
    }

    fn run_window(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        let Some((window_start, window_end, samples)) = self.next_decode_window(is_last) else {
            return Ok(Vec::new());
        };

        let backend = self.backend.as_ref().ok_or(AsrError::NotReady)?;
        let text = backend
            .transcribe_text(&samples, "", self.decode_options)
            .map_err(AsrError::Inference)?;
        self.last_decode_end = window_end;
        let words = transcript_words(&text);
        if words.is_empty() {
            self.buffer.clear();
            self.prune_audio();
            if trace_q5_windows() {
                eprintln!("[asr:q5-window] window={window_start:.1}-{window_end:.1}s empty");
            }
            return Ok(Vec::new());
        }

        let overlap = suffix_prefix_overlap(&self.committed_norm_words, &words);
        if overlap == 0
            && !self.committed_norm_words.is_empty()
            && window_start < self.last_committed_end + 0.25
            && !is_last
        {
            self.buffer = words_to_text(&words);
            self.prune_audio();
            if trace_q5_windows() {
                eprintln!(
                    "[asr:q5-window] window={window_start:.1}-{window_end:.1}s no-overlap committed=0 buffer_words={}",
                    words.len()
                );
            }
            return Ok(Vec::new());
        }

        let commit_hi = if is_last {
            words.len()
        } else {
            words.len().saturating_sub(self.holdback_words)
        };
        if commit_hi <= overlap {
            self.buffer = words_to_text(&words[overlap..]);
            self.prune_audio();
            if trace_q5_windows() {
                eprintln!(
                    "[asr:q5-window] window={window_start:.1}-{window_end:.1}s overlap={overlap} committed=0 buffer_words={}",
                    words.len().saturating_sub(overlap)
                );
            }
            return Ok(Vec::new());
        }

        let mut committed = Vec::new();
        let dur = (window_end - window_start).max(0.001);
        let total = words.len().max(1) as f64;
        let mut j = overlap;
        let mut skipped_dup_words = 0usize;
        let mut skipped_since_last_commit = false;
        while j < commit_hi {
            let dup_len =
                longest_recent_match(&self.committed_norm_words, &words, j).min(commit_hi - j);
            if dup_len >= WINDOWED_Q5_DUP_MIN_WORDS {
                skipped_dup_words += dup_len;
                skipped_since_last_commit = true;
                j += dup_len;
                continue;
            }

            let raw_start = window_start + (j as f64 / total) * dur;
            let raw_end = window_start + ((j as f64 + 1.0) / total) * dur;
            let mut start = raw_start.max(self.last_committed_end);
            if start - self.last_committed_end >= WINDOWED_Q5_GAP_DIAG_SEC {
                let activity =
                    audio_activity_between(&samples, window_start, self.last_committed_end, start);
                let active =
                    activity.rms >= WINDOWED_Q5_ACTIVE_RMS && activity.active_ratio >= 0.08;
                let kind = if active {
                    if skipped_since_last_commit {
                        "dedupe-active"
                    } else {
                        "asr-active-gap"
                    }
                } else {
                    "silence"
                };
                if trace_q5_windows() {
                    eprintln!(
                        "[asr:q5-gap] kind={kind} gap={:.2}s abs={:.1}-{:.1}s rms={:.5} active={:.0}% skipped_dup={}",
                        start - self.last_committed_end,
                        self.last_committed_end,
                        start,
                        activity.rms,
                        activity.active_ratio * 100.0,
                        skipped_dup_words
                    );
                }
                if skipped_since_last_commit && active {
                    start = self.last_committed_end + 0.05;
                }
            }
            let end = raw_end.max(start + 0.01);
            let text = if self.committed_norm_words.is_empty() && committed.is_empty() {
                words[j].text.clone()
            } else {
                format!(" {}", words[j].text)
            };
            committed.push(AsrToken {
                start,
                end,
                text,
                probability: None,
                detected_language: None,
                speaker: None,
            });
            self.committed_norm_words.push(words[j].norm.clone());
            self.last_committed_end = end;
            skipped_since_last_commit = false;
            j += 1;
        }
        self.buffer = words_to_text(&words[commit_hi..]);
        self.prune_audio();
        if trace_q5_windows() {
            eprintln!(
                "[asr:q5-window] window={window_start:.1}-{window_end:.1}s overlap={overlap} committed={} skipped_dup={} buffer_words={} text={:?}",
                committed.len(),
                skipped_dup_words,
                words.len().saturating_sub(commit_hi),
                text.chars().take(120).collect::<String>()
            );
        }
        Ok(committed)
    }

    fn next_decode_window(&self, is_last: bool) -> Option<(f64, f64, Vec<f32>)> {
        if self.audio.is_empty() {
            return None;
        }
        let available_end = self.audio_offset + self.audio.len() as f64 / 16_000.0;
        let scheduled_end = if self.last_decode_end <= 0.0 {
            self.audio_offset + self.min_sec
        } else {
            self.last_decode_end + self.step_sec
        };

        let mut window_end = if scheduled_end <= available_end + 0.001 {
            scheduled_end
        } else if is_last && available_end > self.last_decode_end + 0.25 {
            available_end
        } else {
            return None;
        };
        // 라이브 백로그 상한: 디코드가 max_lag 이상 뒤처지면 최신 윈도우로 점프(중간 구간 누락).
        // 느린 디코드(temperature 폴백/저사양)에도 라이브가 '지금'을 따라가고, 정지 시 잔여
        // 백로그를 길게 토해내지 않게 한다. 따라잡고 있으면(lag<max_lag) 발동하지 않음.
        if !is_last
            && self.max_lag_sec > 0.0
            && available_end - self.last_decode_end > self.max_lag_sec
        {
            window_end = available_end;
        }
        let window_start = (window_end - self.max_sec).max(self.audio_offset);
        if !is_last && window_end - window_start < self.min_sec - 0.001 {
            return None;
        }

        let rel_start = ((window_start - self.audio_offset) * 16_000.0)
            .round()
            .max(0.0) as usize;
        let rel_end = ((window_end - self.audio_offset) * 16_000.0)
            .round()
            .max(rel_start as f64) as usize;
        if rel_end > self.audio.len() || rel_end <= rel_start {
            return None;
        }

        Some((
            window_start,
            window_end,
            self.audio[rel_start..rel_end].to_vec(),
        ))
    }

    fn prune_audio(&mut self) {
        let future_start = if self.last_decode_end > 0.0 {
            self.last_decode_end + self.step_sec - self.max_sec
        } else {
            self.audio_offset
        };
        let keep_from = future_start.max(self.audio_offset);
        let drop_samples = ((keep_from - self.audio_offset) * 16_000.0)
            .floor()
            .max(0.0) as usize;
        if drop_samples > 0 && drop_samples < self.audio.len() {
            self.audio.drain(..drop_samples);
            self.audio_offset += drop_samples as f64 / 16_000.0;
        }

        let max_samples = (self.backlog_sec * 16_000.0) as usize;
        if self.audio.len() > max_samples {
            let extra = self.audio.len() - max_samples;
            self.audio.drain(..extra);
            self.audio_offset += extra as f64 / 16_000.0;
        }
    }
}

#[async_trait]
impl StreamingAsrBackend for WhisperWindowedBackend {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        let path = resolve_ggml(&self.models_dir, &cfg.model_id)?;
        self.decode_options = WhisperDecodeOptions::for_windowed_q5();
        self.backend = Some(
            WhisperRsBackend::load_with_options(&path, None, self.decode_options)
                .map_err(AsrError::Inference)?,
        );
        self.audio.clear();
        self.audio_offset = 0.0;
        self.last_decode_end = 0.0;
        self.buffer.clear();
        self.last_committed_end = 0.0;
        self.committed_norm_words.clear();
        Ok(())
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        let b = self.backend.as_ref().ok_or(AsrError::NotReady)?;
        b.warmup().map_err(AsrError::Inference)
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64) {
        if self.audio.is_empty() {
            self.audio_offset = (end_time - pcm.len() as f64 / 16_000.0).max(0.0);
        }
        self.audio.extend_from_slice(pcm);
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        if !is_last {
            return self.run_window(false);
        }

        // 정지 즉시성: 밀린 백로그를 전부 재전사하지 않는다 — 그게 '정지 후에도 계속 인식되는'
        // 원인이다(느린 디코드로 self.audio 가 최대 backlog_sec 까지 쌓인 뒤 종료 시 한꺼번에 토함).
        // 가장 최근 한 윈도우(≤max_sec)만 확정하고 그 이전 미처리 구간은 버린다.
        let available_end = self.audio_offset + self.audio.len() as f64 / 16_000.0;
        let final_floor = (available_end - self.step_sec).max(0.0);
        if final_floor > self.last_decode_end {
            self.last_decode_end = final_floor;
        }

        let mut out = Vec::new();
        loop {
            let before = self.last_decode_end;
            let mut toks = self.run_window(true)?;
            out.append(&mut toks);
            if self.last_decode_end <= before + 0.001 {
                break;
            }
            let available_end = self.audio_offset + self.audio.len() as f64 / 16_000.0;
            if self.last_decode_end >= available_end - 0.001 {
                break;
            }
        }
        Ok(out)
    }

    fn get_buffer(&self) -> String {
        self.buffer.clone()
    }

    fn set_language(&mut self, _lang: Option<String>) {
        if let Some(b) = &mut self.backend {
            b.set_language(None);
        }
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps {
            provides_word_timestamps: false,
            provides_probability: false,
            self_streaming: true,
            tokenizer_id: "whisper_q5_windowed",
        }
    }
}

fn transcript_words(text: &str) -> Vec<WindowWord> {
    text.split_whitespace()
        .filter_map(|word| {
            let text = word.trim().to_string();
            let norm = normalize_word_for_overlap(word);
            if text.is_empty() || norm.is_empty() {
                None
            } else {
                Some(WindowWord { text, norm })
            }
        })
        .collect()
}

fn normalize_word_for_overlap(word: &str) -> String {
    word.trim_matches(is_noise_punct)
        .trim()
        .to_ascii_lowercase()
}

fn suffix_prefix_overlap(committed_norm_words: &[String], words: &[WindowWord]) -> usize {
    let max = committed_norm_words.len().min(words.len()).min(32);
    for k in (1..=max).rev() {
        let base = committed_norm_words.len() - k;
        if (0..k).all(|i| committed_norm_words[base + i] == words[i].norm) {
            return k;
        }
    }
    0
}

fn longest_recent_match(
    committed_norm_words: &[String],
    words: &[WindowWord],
    word_idx: usize,
) -> usize {
    if word_idx >= words.len() || committed_norm_words.is_empty() {
        return 0;
    }
    let recent_start = committed_norm_words.len().saturating_sub(160);
    let mut best = 0usize;
    for base in recent_start..committed_norm_words.len() {
        let mut n = 0usize;
        while base + n < committed_norm_words.len()
            && word_idx + n < words.len()
            && committed_norm_words[base + n] == words[word_idx + n].norm
        {
            n += 1;
        }
        best = best.max(n);
    }
    best
}

#[derive(Clone, Copy, Debug)]
struct AudioActivity {
    rms: f32,
    active_ratio: f32,
}

fn audio_activity_between(
    samples: &[f32],
    window_start: f64,
    abs_start: f64,
    abs_end: f64,
) -> AudioActivity {
    let start = ((abs_start - window_start).max(0.0) * 16_000.0).floor() as usize;
    let end = ((abs_end - window_start).max(0.0) * 16_000.0).ceil() as usize;
    if start >= end || start >= samples.len() {
        return AudioActivity {
            rms: 0.0,
            active_ratio: 0.0,
        };
    }
    let slice = &samples[start..end.min(samples.len())];
    if slice.is_empty() {
        return AudioActivity {
            rms: 0.0,
            active_ratio: 0.0,
        };
    }

    let rms = (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt();
    let frame = 320usize;
    let mut active = 0usize;
    let mut total = 0usize;
    for chunk in slice.chunks(frame) {
        if chunk.len() < frame / 2 {
            continue;
        }
        total += 1;
        let crms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if crms >= WINDOWED_Q5_ACTIVE_RMS {
            active += 1;
        }
    }
    AudioActivity {
        rms,
        active_ratio: if total == 0 {
            0.0
        } else {
            active as f32 / total as f32
        },
    }
}

fn words_to_text(words: &[WindowWord]) -> String {
    words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn trace_q5_windows() -> bool {
    env_bool("ASR_Q5_TRACE").unwrap_or(false)
}

/// ggml 모델 디렉터리로부터 Whisper StreamingAsrBackend 를 만든다.
pub fn self_streaming_backend(models_dir: impl Into<PathBuf>) -> Box<dyn StreamingAsrBackend> {
    Box::new(WhisperAdaptiveBackend::new(models_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q5_coreml_path_uses_unquantized_turbo_stem() {
        let p = Path::new("/models/ggml-large-v3-turbo-q5_0.bin");
        assert_eq!(
            coreml_encoder_path(p),
            PathBuf::from("/models/ggml-large-v3-turbo-encoder.mlmodelc")
        );
    }

    #[test]
    fn repeated_filler_is_filtered() {
        let text = "아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아, 아";
        assert_eq!(clean_transcript_text(text), "");
    }

    #[test]
    fn short_filler_is_filtered() {
        assert_eq!(clean_transcript_text("아"), "");
        assert_eq!(clean_transcript_text("아, 어"), "");
    }

    #[test]
    fn normal_text_is_kept() {
        let text = "오늘 회의에서는 Q5 모델 성능을 테스트합니다.";
        assert_eq!(clean_transcript_text(text), text);
    }

    #[test]
    fn sentence_with_filler_prefix_is_kept() {
        let text = "아 오늘 회의에서는 성능을 테스트합니다.";
        assert_eq!(clean_transcript_text(text), text);
    }

    #[test]
    fn recent_match_finds_duplicate_phrase() {
        let committed = ["이번", "스프린트", "할", "일", "나누고", "QA랑"]
            .into_iter()
            .map(normalize_word_for_overlap)
            .collect::<Vec<_>>();
        let words = transcript_words("이번 스프린트 할 일 나누고 QA랑 데모");

        assert_eq!(longest_recent_match(&committed, &words, 0), 6);
    }

    #[test]
    fn audio_activity_classifies_active_gap() {
        let samples = vec![0.01; 16_000];
        let activity = audio_activity_between(&samples, 10.0, 10.2, 10.8);

        assert!(activity.rms > WINDOWED_Q5_ACTIVE_RMS);
        assert!(activity.active_ratio > 0.9);
    }
}
