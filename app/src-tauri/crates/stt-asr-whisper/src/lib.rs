//! Rust 네이티브 Whisper ASR — whisper.cpp(whisper-rs, Metal)로 in-process 전사.
//!
//! Python 사이드카를 대체한다. 토큰 단위 타임스탬프를 제공하므로 Rust LocalAgreement
//! (stt-core)의 `WhisperLikeBackend` 로 감싸 스트리밍 확정/partial 을 만든다.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use stt_core::asr::{
    AsrConfig, AsrError, AsrToken, BackendCaps, OnlineAsrProcessor, StreamingAsrBackend,
    WhisperLikeBackend,
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
        if let Some(l) = &self.language {
            params.set_language(Some(l.as_str()));
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

        let n_segments = state.full_n_segments().map_err(|e| e.to_string())?;
        let mut tokens = Vec::new();
        for s in 0..n_segments {
            let n_tok = state.full_n_tokens(s).map_err(|e| e.to_string())?;
            for t in 0..n_tok {
                let text = state
                    .full_get_token_text(s, t)
                    .map_err(|e| e.to_string())?;
                // 특수 토큰([_BEG_], [_TT_...], <|...|>) 제외
                if text.starts_with("[_") || text.starts_with("<|") {
                    continue;
                }
                let data = state.full_get_token_data(s, t).map_err(|e| e.to_string())?;
                // t0/t1 은 centisecond(10ms) 단위
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
    let tmp = path.with_extension("part");
    let mut f = std::fs::File::create(&tmp).map_err(|e| AsrError::Inference(e.to_string()))?;
    std::io::copy(&mut resp.into_reader(), &mut f).map_err(|e| AsrError::Inference(e.to_string()))?;
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

