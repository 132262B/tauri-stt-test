//! Rust 네이티브 Whisper ASR — whisper.cpp(whisper-rs, Metal)로 in-process 전사.
//!
//! Python 사이드카를 대체한다. 토큰 단위 타임스탬프를 제공하므로 Rust LocalAgreement
//! (stt-core)의 `WhisperLikeBackend` 로 감싸 스트리밍 확정/partial 을 만든다.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use stt_core::asr::{
    AsrConfig, AsrError, AsrToken, BackendCaps, OnlineAsrProcessor, SelfStreamingBackend,
    SelfStreamingProcessor, StreamingAsrBackend, WhisperLikeBackend,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// 단어/토큰 단위 전사 결과(절대시각 초).
#[derive(Clone, Debug)]
pub struct WhisperToken {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// whisper.cpp 모델을 적재해 전사를 수행하는 백엔드.
pub struct WhisperRsBackend {
    ctx: WhisperContext,
    language: Option<String>,
}

impl WhisperRsBackend {
    pub fn load(model_path: impl AsRef<Path>, language: Option<String>) -> Result<Self, String> {
        let ctx = WhisperContext::new_with_params(
            model_path.as_ref().to_string_lossy().as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| format!("whisper 모델 적재 실패: {e}"))?;
        Ok(Self { ctx, language })
    }

    pub fn set_language(&mut self, lang: Option<String>) {
        self.language = lang;
    }

    /// 16kHz mono f32 PCM 전체를 전사 → 토큰 리스트(특수토큰 제외).
    pub fn transcribe(&self, audio: &[f32], init_prompt: &str) -> Result<Vec<WhisperToken>, String> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("state 생성 실패: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(4);
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
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, audio)
            .map_err(|e| format!("전사 실패: {e}"))?;

        // whisper-rs 0.16 객체형 API: get_segment → WhisperSegment → get_token.
        let n_segments = state.full_n_segments();
        let mut tokens = Vec::new();
        for s in 0..n_segments {
            let Some(seg) = state.get_segment(s) else { continue };
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
                let Some(tok) = seg.get_token(ti) else { continue };
                let text = match tok.to_str() {
                    Ok(s) => s.to_string(),
                    Err(_) => continue,
                };
                // 특수/주석 토큰 제외([_BEG_], <|..|>, [BLANK_AUDIO] 조각, (..))
                let tt = text.trim_start();
                if tt.starts_with("[_") || tt.starts_with("<|") || tt.starts_with('[') || tt.starts_with('(') {
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
}

impl WhisperLikeBackend for WhisperRsBackend {
    fn transcribe(&self, audio: &[f32], init_prompt: &str) -> Result<Vec<AsrToken>, AsrError> {
        let toks = WhisperRsBackend::transcribe(self, audio, init_prompt)
            .map_err(AsrError::Inference)?;
        Ok(toks
            .into_iter()
            .map(|t| AsrToken {
                start: t.start,
                end: t.end,
                text: t.text,
                probability: None,
                detected_language: None,
                speaker: None,
            })
            .collect())
    }

    fn sep(&self) -> &str {
        ""
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

/// Rust 네이티브 Whisper 스트리밍 백엔드(LocalAgreement 포함). Python 사이드카 불필요.
pub struct WhisperStreamingBackend {
    models_dir: PathBuf,
    language: Option<String>,
    proc: Option<OnlineAsrProcessor<WhisperRsBackend>>,
}

impl WhisperStreamingBackend {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            language: None,
            proc: None,
        }
    }
}

#[async_trait]
impl StreamingAsrBackend for WhisperStreamingBackend {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        self.language = cfg.language.clone();
        let path = resolve_ggml(&self.models_dir, &cfg.model_id)?;
        let backend = WhisperRsBackend::load(&path, cfg.language.clone()).map_err(AsrError::Inference)?;
        self.proc = Some(OnlineAsrProcessor::new(backend, cfg.trimming_sec));
        Ok(())
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        Ok(())
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], _end_time: f64) {
        if let Some(p) = &mut self.proc {
            p.insert_audio_chunk(pcm);
        }
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        let p = self.proc.as_mut().ok_or(AsrError::NotReady)?;
        if is_last {
            Ok(p.finish())
        } else {
            p.process_iter()
        }
    }

    fn get_buffer(&self) -> String {
        self.proc.as_ref().map(|p| p.get_buffer()).unwrap_or_default()
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.language = lang.clone();
        if let Some(p) = &mut self.proc {
            p.backend_mut().set_language(lang);
        }
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps {
            provides_word_timestamps: true,
            provides_probability: false,
            self_streaming: false,
            tokenizer_id: "whisper",
        }
    }
}

/// Whisper 를 발화 단위(전체 텍스트) 백엔드로 노출 — SelfStreamingProcessor 가 연속 2회
/// 전사의 공통 단어 접두사만 확정한다. 토큰 타임스탬프에 의존하지 않으므로 한국어에서
/// 조각/중복/어순 뒤섞임이 없다(LocalAgreement 의 토큰-타임스탬프 의존 문제 회피).
pub struct WhisperSelfBackend {
    models_dir: PathBuf,
    language: Option<String>,
    backend: Option<WhisperRsBackend>,
}

impl WhisperSelfBackend {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            language: None,
            backend: None,
        }
    }
}

impl SelfStreamingBackend for WhisperSelfBackend {
    fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        let path = resolve_ggml(&self.models_dir, &cfg.model_id)?;
        self.language = cfg.language.clone();
        self.backend =
            Some(WhisperRsBackend::load(&path, cfg.language.clone()).map_err(AsrError::Inference)?);
        Ok(())
    }

    fn transcribe_full(&mut self, samples: &[f32], prompt: &str) -> Result<String, AsrError> {
        let b = self.backend.as_ref().ok_or(AsrError::NotReady)?;
        // 직전 확정 텍스트를 whisper initial_prompt 로 전달 → 경계 연속성/정확도 향상.
        let toks = b.transcribe(samples, prompt).map_err(AsrError::Inference)?;
        Ok(toks
            .iter()
            .map(|t| t.text.as_str())
            .collect::<String>()
            .trim()
            .to_string())
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.language = lang.clone();
        if let Some(b) = &mut self.backend {
            b.set_language(lang);
        }
    }
}

/// ggml 모델 디렉터리로부터 SelfStreaming 기반 StreamingAsrBackend 를 만든다.
pub fn self_streaming_backend(models_dir: impl Into<PathBuf>) -> Box<dyn StreamingAsrBackend> {
    Box::new(SelfStreamingProcessor::new(WhisperSelfBackend::new(models_dir)))
}

