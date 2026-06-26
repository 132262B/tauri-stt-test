//! Qwen3-ASR(0.6B) 백엔드 — antirez/qwen-asr(C, MIT)를 in-process FFI 로 호출.
//!
//! 발화 단위 모델(전체 버퍼→텍스트)이라 SenseVoice 와 같이 SelfStreamingProcessor 로
//! 감싸 스트리밍 인터페이스에 맞춘다. Apple Accelerate(BLAS), Python/Node 프로세스 0.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::path::Path;

use asr_core::asr::{
    AsrConfig, AsrError, SelfStreamingBackend, SelfStreamingProcessor, StreamingAsrBackend,
};

extern "C" {
    fn qwen_load(model_dir: *const c_char) -> *mut c_void;
    fn qwen_free(ctx: *mut c_void);
    fn qwen_set_force_language(ctx: *mut c_void, language: *const c_char) -> c_int;
    fn qwen_transcribe_audio(
        ctx: *mut c_void,
        samples: *const c_float,
        n_samples: c_int,
    ) -> *mut c_char;
    fn free(ptr: *mut c_void); // C 가 malloc 한 문자열 해제용(libc)
}

/// Qwen3-ASR 모델 변종(HF repo + 필요한 파일 목록). 0.6B 는 단일, 1.7B 는 분할 safetensors.
pub struct QwenModelSpec {
    pub repo: &'static str,
    pub files: &'static [&'static str],
}

/// Qwen3-ASR-0.6B (~1.7G, 단일 safetensors).
pub const QWEN_06B: QwenModelSpec = QwenModelSpec {
    repo: "Qwen/Qwen3-ASR-0.6B",
    files: &[
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
    ],
};

/// Qwen3-ASR-1.7B (~4.4G, 분할 safetensors + index).
pub const QWEN_17B: QwenModelSpec = QwenModelSpec {
    repo: "Qwen/Qwen3-ASR-1.7B",
    files: &[
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "vocab.json",
        "merges.txt",
    ],
};

/// ISO 코드 → Qwen 이 기대하는 언어명. 미지원/None 은 자동감지.
fn lang_name(lang: &str) -> Option<&'static str> {
    match lang {
        "ko" => Some("Korean"),
        "en" => Some("English"),
        "ja" => Some("Japanese"),
        "zh" => Some("Chinese"),
        _ => None,
    }
}

pub struct QwenBackend {
    ctx: *mut c_void,
}

// C 컨텍스트는 드라이버의 단일 태스크에서만(직렬) 사용되므로 Send 안전.
unsafe impl Send for QwenBackend {}

impl QwenBackend {
    /// model_dir 에 모델 파일이 없으면 spec(HF repo)에서 다운로드 후 적재. language: None=자동.
    pub fn new(
        model_dir: impl AsRef<Path>,
        spec: &QwenModelSpec,
        language: Option<String>,
    ) -> Result<Self, String> {
        let dir = model_dir.as_ref();
        ensure_model(dir, spec)?;
        let c_dir = CString::new(dir.to_string_lossy().as_bytes())
            .map_err(|_| "경로 인코딩 실패".to_string())?;
        let ctx = unsafe { qwen_load(c_dir.as_ptr()) };
        if ctx.is_null() {
            return Err("Qwen3-ASR 모델 적재 실패".into());
        }
        let mut me = Self { ctx };
        me.apply_language(language);
        Ok(me)
    }

    fn apply_language(&mut self, language: Option<String>) {
        match language.as_deref().and_then(lang_name) {
            Some(name) => {
                if let Ok(c) = CString::new(name) {
                    unsafe { qwen_set_force_language(self.ctx, c.as_ptr()) };
                }
            }
            None => unsafe {
                qwen_set_force_language(self.ctx, std::ptr::null());
            },
        }
    }
}

impl Drop for QwenBackend {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { qwen_free(self.ctx) };
        }
    }
}

impl SelfStreamingBackend for QwenBackend {
    fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        self.apply_language(cfg.language.clone());
        Ok(())
    }

    fn transcribe_full(&mut self, samples: &[f32], _prompt: &str) -> Result<String, AsrError> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let ptr =
            unsafe { qwen_transcribe_audio(self.ctx, samples.as_ptr(), samples.len() as c_int) };
        if ptr.is_null() {
            return Err(AsrError::Inference("Qwen3-ASR 전사 실패".into()));
        }
        let text = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { free(ptr as *mut c_void) };
        Ok(text.trim().to_string())
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.apply_language(lang);
    }
}

/// 모델 디렉터리로부터 StreamingAsrBackend(SelfStreamingProcessor 래핑)를 만든다.
pub fn streaming_backend(
    model_dir: impl AsRef<Path>,
    spec: &QwenModelSpec,
    language: Option<String>,
) -> Result<Box<dyn StreamingAsrBackend>, String> {
    let backend = QwenBackend::new(model_dir, spec, language)?;
    Ok(Box::new(SelfStreamingProcessor::new(backend)))
}

/// 필요한 모델 파일이 없으면 spec(HF repo)에서 받아 채운다. 크기 검증 포함.
fn ensure_model(dir: &Path, spec: &QwenModelSpec) -> Result<(), String> {
    std::fs::create_dir_all(dir).ok();
    let base = format!("https://huggingface.co/{}/resolve/main", spec.repo);
    for name in spec.files {
        let name = *name;
        let path = dir.join(name);
        if path.exists() {
            continue;
        }
        let url = format!("{base}/{name}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| format!("{name} 다운로드 실패: {e}"))?;
        let expected: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
        let tmp = path.with_extension("part");
        let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        let written = std::io::copy(&mut resp.into_reader(), &mut f).map_err(|e| e.to_string())?;
        drop(f);
        if let Some(exp) = expected {
            if written != exp {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!(
                    "{name} 다운로드 손상({written}/{exp}B). 재시도 필요."
                ));
            }
        }
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    }
    Ok(())
}
