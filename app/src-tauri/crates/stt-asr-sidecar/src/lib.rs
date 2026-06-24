//! 사이드카 ASR 백엔드 — Python+MLX 사이드카를 spawn 하고 UDS(PCM)/stdin·stdout(NDJSON)로
//! 통신하며 `StreamingAsrBackend` 를 구현한다 (docs/02-architecture.md C·D.7).
//!
//! PCM 은 `insert_audio_chunk`(sync)에서 unbounded 채널로 보내고, 전용 task 가 UDS 로 쓴다.
//! 제어/결과는 stdin/stdout NDJSON. 사이드카 첫 구현이며, 네이티브(mlx-rs/mlx-swift)가 drop-in.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use stt_core::asr::{AsrConfig, AsrError, AsrToken, BackendCaps, StreamingAsrBackend};
use stt_sidecar_proto::{encode_pcm_frame, Control, Event};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 사이드카 실행 방법(개발=venv python, 배포=번들 바이너리).
#[derive(Clone)]
pub struct SidecarSpawn {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub env: Vec<(String, String)>,
}

impl SidecarSpawn {
    /// 개발용: 프로젝트 로컬 venv 의 python -m stt_mlx.main, HF_HOME=프로젝트 캐시.
    pub fn dev_venv(sidecar_dir: impl Into<PathBuf>) -> Self {
        let dir: PathBuf = sidecar_dir.into();
        Self {
            program: dir.join(".venv/bin/python"),
            args: vec!["-m".into(), "stt_mlx.main".into()],
            cwd: dir.clone(),
            env: vec![(
                "HF_HOME".into(),
                dir.join(".hf-cache").to_string_lossy().into_owned(),
            )],
        }
    }
}

struct Inner {
    child: tokio::process::Child,
    control_tx: mpsc::UnboundedSender<Control>,
    pcm_tx: mpsc::UnboundedSender<Vec<u8>>,
    events_rx: mpsc::UnboundedReceiver<Event>,
    tasks: Vec<JoinHandle<()>>,
    uds_path: PathBuf,
}

impl Drop for Inner {
    fn drop(&mut self) {
        for t in &self.tasks {
            t.abort();
        }
        let _ = self.child.start_kill();
        let _ = std::fs::remove_file(&self.uds_path);
    }
}

pub struct SidecarBackend {
    spawn: SidecarSpawn,
    language: Option<String>,
    last_buffer: String,
    inner: Option<Inner>,
}

impl SidecarBackend {
    pub fn new(spawn: SidecarSpawn) -> Self {
        Self {
            spawn,
            language: None,
            last_buffer: String::new(),
            inner: None,
        }
    }

    async fn next_tokens(&mut self) -> Result<(Vec<AsrToken>, String), AsrError> {
        let inner = self.inner.as_mut().ok_or(AsrError::NotReady)?;
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(120), inner.events_rx.recv())
                .await
                .map_err(|_| AsrError::Inference("process_iter 타임아웃".into()))?
                .ok_or(AsrError::BackendDied)?;
            match ev {
                Event::Tokens { committed, buffer, .. } => {
                    let toks = committed
                        .into_iter()
                        .map(|t| AsrToken {
                            start: t.start,
                            end: t.end,
                            text: t.text,
                            probability: t.probability,
                            detected_language: None,
                        })
                        .collect();
                    return Ok((toks, buffer));
                }
                Event::Error { code, msg } => {
                    return Err(AsrError::Inference(format!("{code}: {msg}")))
                }
                // Ready/Warmed/Bye 등은 무시하고 다음 이벤트 대기.
                _ => continue,
            }
        }
    }
}

#[async_trait]
impl StreamingAsrBackend for SidecarBackend {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError> {
        self.language = cfg.language.clone();

        // UDS 경로 준비(Rust 가 bind, 사이드카가 connect).
        let uniq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let uds_path = std::env::temp_dir().join(format!("stt-mlx-{}-{}.sock", std::process::id(), uniq));
        let _ = std::fs::remove_file(&uds_path);
        let listener = UnixListener::bind(&uds_path)
            .map_err(|e| AsrError::Inference(format!("UDS bind 실패: {e}")))?;

        // 사이드카 spawn.
        let mut cmd = Command::new(&self.spawn.program);
        cmd.args(&self.spawn.args)
            .current_dir(&self.spawn.cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        for (k, v) in &self.spawn.env {
            cmd.env(k, v);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| AsrError::Inference(format!("사이드카 spawn 실패: {e}")))?;

        let mut stdin = child.stdin.take().ok_or(AsrError::NotReady)?;
        let stdout = child.stdout.take().ok_or(AsrError::NotReady)?;
        let stderr = child.stderr.take().ok_or(AsrError::NotReady)?;

        let mut tasks = Vec::new();

        // stderr 로깅 task.
        tasks.push(tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("{line}");
            }
        }));

        // stdout NDJSON → events 채널.
        let (events_tx, events_rx) = mpsc::unbounded_channel::<Event>();
        tasks.push(tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<Event>(line) {
                    Ok(ev) => {
                        if events_tx.send(ev).is_err() {
                            break;
                        }
                    }
                    Err(e) => eprintln!("[sidecar-client] 결과 파싱 실패: {e} / {line}"),
                }
            }
        }));

        // 제어(NDJSON) writer task: control 채널 → child stdin.
        let (control_tx, mut control_rx) = mpsc::unbounded_channel::<Control>();
        tasks.push(tokio::spawn(async move {
            while let Some(ctrl) = control_rx.recv().await {
                match ctrl.to_ndjson() {
                    Ok(line) => {
                        if stdin.write_all(line.as_bytes()).await.is_err() {
                            break;
                        }
                        let _ = stdin.flush().await;
                    }
                    Err(e) => eprintln!("[sidecar-client] 제어 직렬화 실패: {e}"),
                }
            }
        }));

        // configure 전송.
        control_tx
            .send(Control::Configure {
                model: cfg.model_id.clone(),
                lang: cfg.language.clone(),
                uds_path: uds_path.to_string_lossy().into_owned(),
                trimming_sec: cfg.trimming_sec,
            })
            .map_err(|_| AsrError::BackendDied)?;

        // 사이드카의 UDS connect 수락 → PCM writer task.
        let (pcm_tx, mut pcm_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (stream, _addr) = tokio::time::timeout(Duration::from_secs(30), listener.accept())
            .await
            .map_err(|_| AsrError::Inference("사이드카 UDS 연결 타임아웃".into()))?
            .map_err(|e| AsrError::Inference(format!("UDS accept 실패: {e}")))?;
        tasks.push(tokio::spawn(async move {
            let mut stream = stream;
            while let Some(frame) = pcm_rx.recv().await {
                if stream.write_all(&frame).await.is_err() {
                    break;
                }
            }
        }));

        self.inner = Some(Inner {
            child,
            control_tx,
            pcm_tx,
            events_rx,
            tasks,
            uds_path,
        });

        // ready 대기.
        let inner = self.inner.as_mut().unwrap();
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(180), inner.events_rx.recv())
                .await
                .map_err(|_| AsrError::Inference("ready 타임아웃".into()))?
                .ok_or(AsrError::BackendDied)?;
            match ev {
                Event::Ready { .. } => return Ok(()),
                Event::Error { code, msg } => {
                    return Err(AsrError::Inference(format!("configure: {code}: {msg}")))
                }
                _ => continue,
            }
        }
    }

    async fn warmup(&mut self) -> Result<(), AsrError> {
        let inner = self.inner.as_mut().ok_or(AsrError::NotReady)?;
        inner
            .control_tx
            .send(Control::Warmup)
            .map_err(|_| AsrError::BackendDied)?;
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(120), inner.events_rx.recv())
                .await
                .map_err(|_| AsrError::Inference("warmup 타임아웃".into()))?
                .ok_or(AsrError::BackendDied)?;
            match ev {
                Event::Warmed => return Ok(()),
                Event::Error { code, msg } => {
                    return Err(AsrError::Inference(format!("warmup: {code}: {msg}")))
                }
                _ => continue,
            }
        }
    }

    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64) {
        if let Some(inner) = &self.inner {
            let _ = inner.pcm_tx.send(encode_pcm_frame(pcm, end_time));
        }
    }

    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError> {
        {
            let inner = self.inner.as_ref().ok_or(AsrError::NotReady)?;
            let ctrl = if is_last { Control::Finish } else { Control::ProcessIter };
            inner.control_tx.send(ctrl).map_err(|_| AsrError::BackendDied)?;
        }
        let (tokens, buffer) = self.next_tokens().await?;
        self.last_buffer = buffer;
        Ok(tokens)
    }

    fn get_buffer(&self) -> String {
        self.last_buffer.clone()
    }

    fn set_language(&mut self, lang: Option<String>) {
        self.language = lang.clone();
        if let Some(inner) = &self.inner {
            let _ = inner.control_tx.send(Control::SetLanguage { lang });
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
