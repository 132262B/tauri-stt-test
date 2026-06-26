//! 모델 저장소 — 런타임 경로 해석 + 클라우드 다운로드(진행률 emit).
//!
//! 기존엔 모델 루트가 컴파일타임 `env!(CARGO_MANIFEST_DIR)/models` 였다. 이 경로는
//! 패키지된 번들(.app/.ipa/.apk/.msi)에는 존재하지 않아 모델 로드가 실패했다(크로스플랫폼
//! 출시 블로커). 여기서는 런타임에 **쓰기 가능한** 위치(app_data_dir)로 해석하고,
//! 누락 모델을 HF 에서 받아 그 곳에 채운다(다운로드는 read-only 번들 디렉터리가 아닌
//! app_data_dir 로 향한다).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

/// 모델 루트 디렉터리(쓰기 가능). 우선순위:
/// 1) `STT_MODELS_DIR` 환경변수(명시 override)
/// 2) (디버그 빌드) 인-트리 `CARGO_MANIFEST_DIR/models` 가 있으면 — 개발 중 수 GB 재다운로드 회피
/// 3) `app_data_dir/models` — 패키지 빌드의 정식 위치(쓰기 가능, 다운로드가 여기로 떨어진다)
pub fn models_root(app: &AppHandle) -> PathBuf {
    if let Ok(p) = std::env::var("STT_MODELS_DIR") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    #[cfg(debug_assertions)]
    {
        let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        if dev.exists() {
            return dev;
        }
    }
    let dir = app
        .path()
        .app_data_dir()
        .map(|d| d.join("models"))
        .unwrap_or_else(|_| PathBuf::from("models"));
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// 번들/개발 데이터 파일(예: 데모 wav)의 런타임 경로. 디버그는 인-트리, 릴리스는 리소스 디렉터리.
pub fn data_file(app: &AppHandle, rel: &str) -> PathBuf {
    #[cfg(debug_assertions)]
    {
        let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
        if dev.exists() {
            return dev;
        }
    }
    app.path()
        .resource_dir()
        .map(|d| d.join(rel))
        .unwrap_or_else(|_| PathBuf::from(rel))
}

// ── 카탈로그 ──────────────────────────────────────────────────────────────

/// 다운로드 가능한 단일 파일(HF resolve URL → 모델 루트 기준 상대경로).
struct ModelFile {
    url: &'static str,
    rel: &'static str,
    /// 검증용 기대 바이트(0=Content-Length 로 검증).
    bytes: u64,
}

struct ModelEntry {
    id: &'static str,
    label: &'static str,
    /// 존재 판별용 대표 파일(모델 루트 기준 상대경로).
    probe: &'static str,
    approx_mb: u32,
    /// HF 자동 다운로드 파일 목록(비면 수동 배치 필요 = downloadable=false).
    files: &'static [ModelFile],
}

const HF_WHISPER: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const HF_QWEN: &str = "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main";

const CATALOG: &[ModelEntry] = &[
    ModelEntry {
        id: "ggml-large-v3-turbo-q5_0",
        label: "Whisper large-v3-turbo Q5 (기본)",
        probe: "ggml/ggml-large-v3-turbo-q5_0.bin",
        approx_mb: 547,
        files: &[ModelFile {
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
            rel: "ggml/ggml-large-v3-turbo-q5_0.bin",
            bytes: 574_041_195,
        }],
    },
    ModelEntry {
        id: "qwen",
        label: "Qwen3-ASR 0.6B",
        probe: "qwen/model.safetensors",
        approx_mb: 1789,
        files: &[
            ModelFile { url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/config.json", rel: "qwen/config.json", bytes: 0 },
            ModelFile { url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/generation_config.json", rel: "qwen/generation_config.json", bytes: 0 },
            ModelFile { url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/model.safetensors", rel: "qwen/model.safetensors", bytes: 1_876_091_704 },
            ModelFile { url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/vocab.json", rel: "qwen/vocab.json", bytes: 0 },
            ModelFile { url: "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/merges.txt", rel: "qwen/merges.txt", bytes: 0 },
        ],
    },
    // SenseVoice / 화자분리(pyannote+CAM++)는 sherpa-onnx 릴리스 자산이라 자동 다운로드 URL 을
    // 아직 고정하지 않는다(수동 배치). 상태(present/missing)만 보고한다.
    ModelEntry {
        id: "sensevoice",
        label: "SenseVoice 다국어",
        probe: "sense/model.onnx",
        approx_mb: 228,
        files: &[],
    },
    ModelEntry {
        id: "diar",
        label: "화자분리(pyannote + CAM++)",
        probe: "diar/sherpa-onnx-pyannote-segmentation-3-0",
        approx_mb: 35,
        files: &[],
    },
];

// 미사용 경고 억제(HF_WHISPER/HF_QWEN 는 가독성용 상수 — 카탈로그는 전체 URL 을 직접 보관).
#[allow(dead_code)]
const _BASES: (&str, &str) = (HF_WHISPER, HF_QWEN);

// ── 상태/다운로드 ─────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: String,
    pub label: String,
    pub present: bool,
    pub approx_mb: u32,
    /// HF 자동 다운로드 가능 여부(false=수동 배치 필요).
    pub downloadable: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DownloadProgress<'a> {
    model: &'a str,
    file: &'a str,
    received: u64,
    total: u64,
    /// 0.0~1.0 (total 미상이면 -1.0).
    pct: f32,
}

/// 모델별 설치 상태 목록.
pub fn status(app: &AppHandle) -> Vec<ModelStatus> {
    let root = models_root(app);
    CATALOG
        .iter()
        .map(|e| ModelStatus {
            id: e.id.to_string(),
            label: e.label.to_string(),
            present: root.join(e.probe).exists(),
            approx_mb: e.approx_mb,
            downloadable: !e.files.is_empty(),
        })
        .collect()
}

/// 모델 하나를 다운로드(이미 있으면 즉시 성공). 파일별 진행률을 `model_download_progress` 로 emit.
pub fn download(app: &AppHandle, id: &str) -> Result<(), String> {
    let entry = CATALOG
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| format!("알 수 없는 모델: {id}"))?;
    if entry.files.is_empty() {
        return Err(format!(
            "{} 은(는) 자동 다운로드 미지원입니다(수동 배치 필요).",
            entry.label
        ));
    }
    let root = models_root(app);
    for f in entry.files {
        download_file(app, entry.id, f, &root)?;
    }
    Ok(())
}

fn download_file(app: &AppHandle, model_id: &str, f: &ModelFile, root: &Path) -> Result<(), String> {
    let dest = root.join(f.rel);
    if dest.exists() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", f.rel))?;
    }
    let resp = ureq::get(f.url)
        .call()
        .map_err(|e| format!("{} 다운로드 실패: {e}", f.rel))?;
    let total: u64 = resp
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(f.bytes);
    let tmp = dest.with_extension("part");
    let mut out = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut received: u64 = 0;
    let mut last_emit: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        received += n as u64;
        // ~4MB 마다(또는 완료 시) 진행률 emit — 이벤트 폭주 방지.
        if received - last_emit >= 4_000_000 || (total > 0 && received >= total) {
            last_emit = received;
            let pct = if total > 0 {
                (received as f32 / total as f32).clamp(0.0, 1.0)
            } else {
                -1.0
            };
            let _ = app.emit(
                "model_download_progress",
                DownloadProgress { model: model_id, file: f.rel, received, total, pct },
            );
        }
    }
    drop(out);
    // 잘린 다운로드는 깨진 모델 → whisper.cpp/sherpa 가 abort 하므로 크기로 무결성 검증.
    if total > 0 && received != total {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "{} 다운로드 손상: {received}/{total}B. 다시 시도하세요.",
            f.rel
        ));
    }
    std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
    Ok(())
}
