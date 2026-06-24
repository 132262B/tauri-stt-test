# 02 — 확정 아키텍처 (온디바이스 회의 전사 Tauri 앱)

> 출처/전제: `00-product-seed.md`(canonical 사양) + `01-whisper-live-kit-analysis.md`(file:line 근거·포팅표·trait 초안·tokio 매핑·오픈 퀘스천) + 확정 결정 6항.
> 이 문서는 제안 A(리스크 우선/수직 슬라이스), B(장기 구조/iOS 대비), C(Tauri 관용/스캐폴드 통합) 세 안과 3건의 심사를 종합한 **단일 확정안**이다. 베이스 = **제안 C**(Tauri 2 관용 + 검증된 스캐폴드 활용), 여기에 **B의 crate 격리·async trait·BackendCaps·CI 불변식**과 **A의 리스크 번다운 커밋 시퀀싱·크래시 복구 가드·[REFACTOR] 부채 태깅**을 접목하고, 세 심사가 공통 지적한 사각지대(화자 라벨 reconciliation, hot-swap 전사 정합, 결정2 vs 자체스트리밍 모순, 한·영 intra-sentence 코드스위칭, mic+system 믹싱/동기, 골든 회귀 테스트, 사이드카 공증, Metal/GPU 메모리 관측)를 본문에 해소한다.
> 표기 규칙: 근거가 약하면 **[가정]**, 미검증 위험은 **[확인 필요]**(분석문서 오픈 퀘스천 승계), 동작 우선 부채는 **[REFACTOR Rn]**.

---

## 0. 설계 원칙 (관통 규칙)

1. **사이드카 우선, 정책/텐서 분리**(확정결정 1). 1단계 Mac에서 Python+MLX 사이드카로 동작 확보 → 2단계 스트리밍 정책/상태를 Rust로 이전(텐서연산만 MLX) → 3단계 iOS는 mlx-swift 네이티브. 사이드카는 iOS 불가.
2. **스트리밍 정책 = LocalAgreement-2 단일화**(확정결정 2). AlignAtt/cross-attention 커스텀 디코더(`align_att_base.py:534`의 ~20 abstractmethod, 5.2 `AlignAttHooks`)는 **설계에서 전면 배제** → 분석문서 오픈Q2 자동 해소. (단, "정책 단일화"의 정확한 의미는 D.5에서 자체스트리밍 백엔드와의 관계로 확정한다 — 세 심사가 공통 지적한 모순 지점.)
3. **플랫폼 의존을 trait 뒤로 격리**(B 관점). 캡처·추론·화자분리 모델만 플랫폼/백엔드 종속, 나머지(정책·파이프라인·출력·모델매니저·메트릭·config)는 플랫폼 무관 순수 Rust. iOS 전환 시 `#[cfg]`+feature 면적을 최소화.
4. **FFmpeg 미사용**(확정결정 4). 네이티브 캡처(cpal/AVAudioEngine, ScreenCaptureKit)가 16kHz mono f32 PCM을 직접 공급(원본 `pcm_input=True` 경로, `audio_processor.py:753`). 리샘플은 anti-alias 없는 단순평균(`recorder_worker.js:27-48`) 대신 polyphase(`rubato`).
5. **클라우드 0을 코드 레벨로 보장**(확정결정 5, ontology "오프라인성"). 다운로드 단계와 추론 단계를 물리적으로 분리하고, 추론 crate 의존 그래프에 HTTP 클라이언트가 없음을 CI grep으로 단언(B). 원본은 오프라인 강제 플래그가 없다(`HF_HUB_OFFLINE` grep 0건, 분석 6.1).
6. **작업 단위마다 커밋, 커밋 메타에 AI 도구 표기 금지**(seed Constraint 8·9). 워크플로 규칙이며 설계 문서엔 영향 없음.

---

## A. 크레이트 / 모듈 구조

### A.1 결론: `src-tauri/Cargo.toml` 내부 워크스페이스 + 순수 코어 crate 분리

기존 `[lib] name="app_lib"` / `crate-type=["staticlib","cdylib","rlib"]` 와 `#[cfg_attr(mobile, tauri::mobile_entry_point)] pub fn run()`(현재 `src-tauri/src/lib.rs`에 실재)는 cargo-mobile이 iOS entry point를 거는 지점이므로 **이름·crate-type·entry point를 그대로 유지**한다. 워크스페이스는 **`app/` 루트가 아니라 `src-tauri/Cargo.toml`에 `[workspace]`로 선언**한다 — Tauri/cargo-mobile이 `src-tauri`에서 cargo를 호출하므로 manifest 루트를 흔들지 않는 배치가 마찰이 가장 적다(심사 검증: `gen/apple/project.yml`의 preBuildScript는 `$SRCROOT/Externals/{arm64,x86_64}/${CONFIGURATION}/libapp.a`를 산출, B의 `app/` 루트 별도 manifest보다 우월).

```
app/src-tauri/                       # 위치 유지 (cargo/tauri 루트)
├── Cargo.toml                       # [workspace] members=[".","crates/*"] 추가 + 어댑터 deps
├── build.rs                         # 유지
├── tauri.conf.json                  # externalBin/resources/플러그인 등록 (K)
├── capabilities/{default.json,mobile.json}  # 데스크톱/모바일 권한 분리 (E.3)
├── resources/{model_catalog.json,warmup_16k.f32}  # 번들 카탈로그 + 오프라인 warmup PCM
├── binaries/stt-mlx-sidecar-<triple>           # PyInstaller 사이드카(빌드 산출물, Mac, gitignore)
├── sidecar/                         # Python 사이드카 소스 (C) — Rust 워크스페이스 밖
│   └── stt_mlx/{main.py, framing.py, online_asr.py, mlx_backend.py}
└── src/                             # = app_lib : 얇은 Tauri 어댑터 (tauri만 의존)
    ├── main.rs                      # 유지 (app_lib::run())
    ├── lib.rs                       # Builder 조립 + .manage(AppState) + invoke_handler + emit 브리지
    ├── commands.rs                  # #[tauri::command] (E.2)
    ├── events.rs                    # Rust→프론트 serde 페이로드 + emit 헬퍼 (E.1)
    ├── app_state.rs                 # AppState: SessionHandle, ModelManager, 백엔드 선택
    └── capture/                     # 플랫폼 의존 캡처 (cfg 격리) — tauri State에서 spawn
        ├── mod.rs                   #  CaptureSource trait, AudioFrame{samples,t_start,t_end,source}
        ├── mic_cpal.rs             #  #[cfg(desktop)] 마이크 (1단계)
        ├── mic_avaudio.rs          #  #[cfg(target_os="ios")] AVAudioEngine FFI (4단계)
        ├── screencapturekit.rs     #  #[cfg(target_os="macos")] 시스템오디오
        ├── mixer.rs                #  mic+system 2소스 정렬/믹싱 (B.6, 신규 해소)
        └── resample.rs             #  rubato polyphase → 16k mono f32
```

```
app/src-tauri/crates/
├── stt-core/                        # 플랫폼·tauri 무의존 순수 Rust (단위테스트·iOS 재사용 핵심)
│   └── src/
│       ├── lib.rs
│       ├── audio.rs                 # PcmFrame(16k mono f32 + 절대시각), 링버퍼
│       ├── vad/                     # Silero VAC: 512프레임 청커 + 히스테리시스 상태머신
│       │   ├── mod.rs               #  Vad trait
│       │   └── silero.rs            #  ort 세션 + _state(2,1,128)+64ctx + threshold 0.5/0.35
│       ├── asr/
│       │   ├── backend.rs           #  StreamingAsrBackend / WhisperLikeBackend trait + BackendCaps (D)
│       │   ├── token.rs             #  AsrToken / AsrResult / AsrError
│       │   ├── policy/              #  LocalAgreement-2 (정책, 플랫폼무관)
│       │   │   ├── local_agreement.rs  #  OnlineAsrProcessor + HypothesisBuffer
│       │   │   ├── trimming.rs      #  segment 트리밍(한국어 강제) + 프롬프트 캐리오버 200자
│       │   │   └── utf8_pending.rs  #  멀티바이트 부분토큰 `�` pending (한국어 필수, 4.x)
│       │   └── registry.rs          #  BackendKind enum + AsrBackendFactory
│       ├── diar/
│       │   ├── mod.rs               #  OnlineDiarBackend trait, SpeakerSegment
│       │   ├── onnx_diar.rs         #  segmentation+embedding (ort)
│       │   ├── tracker.rs           #  청크 간 화자 동일성 유지(spkcache/fifo 대체) + reconciliation (I.1)
│       │   ├── enroll.rs            #  화자 등록/식별 (임베딩 추출)
│       │   ├── gallery.rs           #  SpeakerGallery (SQLite) + cosine 매칭
│       │   └── alignment.rs         #  토큰⨉화자⨉침묵 정렬(tokens_alignment.py port) + 클램핑 (B.5/오픈Q8)
│       ├── pipeline/
│       │   ├── mod.rs               #  PipelineEvent / 채널 토폴로지 spawn
│       │   ├── event.rs             #  PcmEvent(내부) vs PipelineEvent(emit) 2계층 (B.2)
│       │   ├── state_actor.rs       #  State 단일 소유 actor (asyncio.Lock 대체)
│       │   └── backpressure.rs      #  bounded mpsc + 경계 병합 + coalesce (B.4)
│       ├── output/
│       │   ├── snapshot.rs          #  TranscriptSnapshot/Line(ts=f64, speaker=enum) (E.1)
│       │   ├── diff.rs              #  full/diff + 화자 소급 정정(relabel) 계산 (E.1, I.1)
│       │   └── export.rs            #  txt/srt/json writer (seed C11)
│       ├── models/
│       │   ├── catalog.rs           #  정적 카탈로그 + commit SHA 핀
│       │   ├── downloader.rs        #  hf-hub/reqwest, 진행률 (desktop feature 전용)
│       │   ├── cache.rs             #  app_data_dir 단일 캐시 + sha256 무결성 + 용량/축출 (G)
│       │   └── offline.rs           #  오프라인 강제 게이트 (로컬 경로만, 미존재=ModelMissing)
│       ├── metrics/
│       │   ├── mod.rs               #  ResourceMonitor (1초 emit)
│       │   ├── system.rs            #  sysinfo/mach task_info (CPU/RSS, 사이드카 PID 합산)
│       │   └── pipeline_metrics.rs  #  rtf/지연 p50·p95/queue depth (metrics_collector port)
│       └── config.rs                #  AppConfig serde + validate(.en→lang 등)
├── stt-sidecar-proto/               # 사이드카 프레임/메시지 스키마 (Rust↔Python 공유 계약, C.2)
│   └── src/lib.rs                   #  serde 구조체 only. Python은 동일 필드 미러.
└── stt-asr-sidecar/                 # #[cfg(all(macos,feature="sidecar"))] StreamingAsrBackend 첫 구현
    └── src/lib.rs                   #  프로세스 관리 + proto + UDS PCM 전송 (C)
```

### A.2 책임 불변식 (B의 핵심 이득)

- `stt-core`는 **`tauri`를 절대 import 하지 않는다.** 외부와는 (a) 입력 `mpsc<PcmFrame>`, (b) 출력 `broadcast<PipelineEvent>` 두 채널로만 접한다 → 헤드리스 통합테스트(녹음 WAV → 기대 이벤트 시퀀스, J 회귀 스위트)와 iOS 재사용이 동시에 성립.
- `app_lib`(`src-tauri/src`)는 **어댑터 셸**: core의 `broadcast<PipelineEvent>`를 구독해 `app.emit()`으로 옮기고, 프론트 command를 core API 호출로 변환할 뿐.
- `capture`는 플랫폼 의존이라 `app_lib` 측에 두고 core엔 `CaptureSource` trait만(또는 `crates/stt-capture`로 추출, 후순위). core는 PCM 프레임만 받는다.
- **CI 불변식**(클라우드 0): `cargo tree -p stt-core --no-default-features`에 `reqwest`/`hf-hub`/HTTP 클라이언트가 없음을 CI에서 grep 단언. 다운로드는 `models` 모듈의 `desktop` feature에만 존재.

---

## B. 라이브 파이프라인 (tokio)

### B.1 데이터 흐름

```
capture(native)        vad(core,순수)       asr(trait, MLX/사이드카 뒤)   diar(trait,onnx)       output(core)→app_lib
─────────────          ──────────           ─────────────────────────     ─────────────          ────────────────────
cpal/SCK/AVAudio   ─▶  SileroVac       ─▶   StreamingAsrBackend       ─▶  State actor       ─▶   diff/relabel 계산 → emit
mixer 16k mono         512프레임 게이트       (LocalAgreement-2 정책 /      (tokens⨉speakers       transcript_update
+ rubato               PcmEvent fan-out      자체스트리밍)                  ⨉silence 정렬,        metrics_update
                                             process_iter→확정             reconciliation)        transcript_done
                                             get_buffer→partial            OnlineDiarBackend
```

### B.2 PipelineEvent — 내부 dispatch / IPC emit 2계층 (C 채택)

내부 enum(캡처→정책 dispatch)과 프론트 emit 페이로드를 분리해 결합도를 낮춘다.

```rust
// crates/stt-core/src/pipeline/event.rs

/// VAD가 생성 → ASR/diar 양 채널 fan-out. 원본 isinstance 분기(audio_processor.py:382-401) 치환.
pub enum PcmEvent {
    Pcm { samples: std::sync::Arc<[f32]>, t_start: f64, t_end: f64 }, // 활성 슬라이스(절대시각, Arc로 zero-copy 복제)
    SilenceStart { at: f64 },                  // 원본 _begin_silence
    SilenceEnd   { at: f64, duration: f64 },   // 원본 _end_silence
    ChangeSpeaker { at: f64 },                 // diar→ASR 컨텍스트 리셋 훅
    Sentinel,                                  // end-of-stream (원본 SENTINEL)
}

/// 프론트로 emit 되는 통합 이벤트(E.1 IPC 스키마와 1:1). 내부 PcmEvent와 분리.
pub enum PipelineEvent {
    TranscriptUpdate(TranscriptSnapshot),   // added/updated/relabeled/buffer
    TranscriptDone(SessionSummary),
    Metrics(MetricsSnapshot),
    ModelDownloadProgress(DownloadProgress),
    BackendState(BackendStateEvent),         // 로딩/Ready/Switching/Error
    CaptureState(CaptureStateEvent),         // overrun/permission
    Error(PipelineError),
}
```

### B.3 태스크/채널 토폴로지 (tokio)

분석 4.2 매핑표 승계. 단일 사용자라 전역 모델 락(`thread_safety.py:34`)은 제거하고 MLX는 단일 워커에 직렬화.

```
[capture task]  cpal/SCK/AVAudio 콜백 → mixer(B.6) → rubato 16k mono
   │ mpsc<PcmFrame>  bounded(≈2s)         ── 백프레셔 ① (오디오 손실 금지)
   ▼
[vad task]  SileroVac (spawn_blocking 또는 전용 스레드의 ort)
   │  512프레임 게이트 + 히스테리시스 → PcmEvent
   ├──▶ mpsc<PcmEvent> transcription_tx   bounded ── 백프레셔 ②
   └──▶ mpsc<PcmEvent> diarization_tx     bounded
            │ (사이드카=async IPC / 네이티브=MLX 워커 액터 oneshot)   │ (ort, spawn_blocking)
   [asr task] StreamingAsrBackend.process_iter()        [diar task] OnlineDiarBackend.accept_pcm()
            │ mpsc<StateMsg::Asr{committed,buffer}>      │ mpsc<StateMsg::Diar{segments}>
            └───────────────────┬─────────────────────────┘
                                ▼
              [state actor task]  State 단독 소유 (락 없음)
                alignment::update(tokens⨉speakers⨉silence) + reconciliation(I.1)
                → TranscriptSnapshot(added/updated/relabeled) diff 계산
                                │ broadcast(coalesce, 최신우선) ── 백프레셔 ③
                                ▼
              [emit bridge task] (app_lib) → app.emit("transcript_update", …)
              [metrics task] (1s interval) → app.emit("metrics_update", …)   (독립 루프)
```

세션당 태스크: capture / vad / asr / diar / state-actor / emit-bridge / metrics = 7. 세션 종료는 `tokio_util::sync::CancellationToken` 하나로 일괄 취소. MLX는 분석 4.2/`qwen3_vllm_metal_asr.py:55-107` 근거로 **전용 스레드+요청 채널 액터**에 직렬화(사이드카 단계는 프로세스 1개가 곧 직렬화 지점).

### B.4 백프레셔 (분석 4.3 충실)

| 채널 | 정책 |
|---|---|
| ① capture→vad, ② vad→asr/diar | **오디오 손실 금지.** bounded mpsc. 가득 차면 `try_recv` 루프로 active 슬라이스를 **Silence/Sentinel 경계까지만** concat 병합(`get_all_from_queue` 의미를 clean 재구현, **사적 `_queue` 접근 금지**). 경계 깨면 타임스탬프 드리프트. 병합 상한(예 30s)은 P1 튜닝 **[REFACTOR R-bp]**. |
| ③ state→emit | **최신 우선.** `broadcast`(또는 `watch`)로 마지막 스냅샷만 유지(coalesce). 단, **relabel 이벤트는 손실되면 안 되므로**(I.1 소급 정정) diff에 `relabeled` 필드를 누적 포함시켜 coalesce에서도 유실되지 않게 한다. |

### B.5 타임스탬프 정합 (최상위 위험, 오픈Q8)

producer sample-precise total ⨉ consumer cumulative stream time 분리 추적 + 클램핑(`audio_processor.py:798-810`). `tokens_alignment.py`는 분석 범위 밖 → **P1에서 별도 정밀 분석 후 `diar/alignment.rs`에 이식**. **P0(수직 슬라이스)에서는 단어 ts pass-through**(드리프트 감수)로 격리하여 PoC를 멈추지 않게 한다(A의 리스크 격리). 긴 회의(>30분) 정확도는 P1 정합 완료까지 acceptance 보류.

### B.6 mic + system 2소스 동시 입력 (세 심사 공통 사각지대 해소)

seed C7/AC는 Mac에서 마이크+시스템오디오 동시를 요구한다. `capture/mixer.rs`:
- **클록 드리프트**: 두 소스를 각자 `AudioFrame{t_start,t_end,source}`(소스별 단조 타임라인)로 받아, **공통 세션 클록(캡처 시작 시각 기준 monotonic)** 으로 리스탬프. 두 디바이스 샘플레이트 차이는 각각 rubato로 16k 정규화 후 정렬.
- **합류 정책(확정)**: 화자분리 품질을 위해 **2채널을 분리 유지하지 않고 시각 정렬 후 단일 mono로 믹스**(가산 평균, 클리핑 가드)하여 VAD/ASR/diar에 하나의 스트림으로 공급. 이유: 온라인 diar이 동일 화자를 채널 무관하게 클러스터링해야 하고, 라이브 단일 타임라인 전사가 산출물이기 때문. 소스 태그(`mic`/`system`)는 프레임 메타로 보존해 진단/지연측정에만 사용. **[가정]** 에코(시스템 출력이 마이크로 재유입) 억제는 P1.5에서 간단한 정렬-차감 또는 소스 우선순위로 완화 **[확인 필요]**.

---

## C. 사이드카 프로토콜 (1단계, Mac 전용)

### C.1 프로세스 관리: Tauri shell sidecar(externalBin) 사용

확정결정 1 + Tauri 관용. **`tauri-plugin-shell`의 sidecar(externalBin)** 로 PyInstaller 단일 실행파일을 번들·서명한다. `bundle.externalBin: ["binaries/stt-mlx-sidecar"]` → Tauri가 타깃 트리플 접미사로 번들. iOS는 externalBin 자체가 불가 → "사이드카 iOS 불가"와 정합. 단, **프로세스 spawn/IO는 `stt-asr-sidecar` crate가 직접 관리**하고 core는 trait만 본다.

### C.2 메시지 프레이밍: 제어/결과 = stdout NDJSON, PCM = Unix Domain Socket (C의 베스트 아이디어 채택)

세 심사가 꼽은 핵심: **stdout에 바이너리 PCM 프레임을 흘리면 Tauri `CommandEvent::Stdout`의 라인 버퍼링/UTF-8 가정과 충돌**(제안 A의 단일 stdin 바이너리 방식 위험). 따라서 **비대칭 분리**:

- **제어(Rust→py) = stdin NDJSON** + **결과(py→Rust) = stdout NDJSON** + **로그 = stderr.**
  - stdin: `{"type":"configure","backend":"mlx_whisper","model_path":"...","lang":null}`, `{"type":"process_iter","is_last":false}`, `{"type":"set_language","lang":"auto"}`, `{"type":"change_speaker","at":12.3}`, `{"type":"finish"}`, `{"type":"warmup"}`.
  - stdout: `{"type":"tokens","committed":[{start,end,text,prob?,lang?}],"buffer":"...","is_final":bool,"lag_sec":f64}`, `{"type":"ready","backend":"...","model":"...","sr":16000}`, `{"type":"error","code":"...","msg":"..."}`. **stdout은 프레임 전용** — 라이브러리 print 오염 방지를 위해 부트 시 `sys.stdout`을 프레임 writer로 교체하고 mlx/transformers logger를 stderr로 redirect.
- **PCM 전송 = Unix Domain Socket**(`tokio::net::UnixStream`). `configure` 시 Rust가 `$TMPDIR`에 UDS 경로를 만들어 알려주면 사이드카가 connect. 프레임 = `u32 LE length ‖ f32 LE PCM(+ 8B f64 t_end)`. 이점: (a) base64 팽창 0, (b) stdin 제어/PCM 인터리빙 프레이밍 난점 회피, (c) **백프레셔가 OS 소켓버퍼로 자연 전파**(가득 차면 write 블록 → B.4의 ② 정책으로 전파). 스키마는 `stt-sidecar-proto` crate(serde)에 단일 정의, Python은 동일 필드 미러.

### C.3 생명주기 / 크래시 복구 (A의 폭주 가드 채택)

- **기동**: `start_session`/`select_backend` 시 lazy spawn → `ready` 수신까지 await(타임아웃 10s, 초과 시 `Error(SidecarStartTimeout)`) → `warmup`(번들 PCM, G) → 스트리밍.
- **종료**: `finish` → 잔여 flush 수신 → stdin EOF → grace → 타임아웃 후 SIGKILL. 고아 방지: `app.cleanup_before_exit` 훅 + RAII Drop에서 명시 kill.
- **크래시 복구**: `CommandEvent::Terminated`/stdout EOF 감지 → `Error(SidecarCrashed)` emit + **확정분 보존** + **자동 재기동(backoff)**. PCM 백로그는 폐기(라이브 연속성 포기, VAD 상태는 Rust 보유라 무관). **재기동 폭주 방지: 60초 내 3회 초과 시 영구 실패 처리**(MLX OOM/세그폴트 대비, A의 정량 가드 — B/C의 "1회 재spawn"보다 구체적).
- **백프레셔**: UDS write 블록 → asr task `process_iter` 지연 → 채널 ② 차오름 → B.4 발동. 사이드카가 `lag_sec`를 RESULT에 실어 UI "처리 지연" 경고로 노출.

---

## D. 교체형 ASR trait (확정 시그니처)

분석 5.1 초안(StreamingAsrBackend/WhisperLikeBackend)을 확정. **5.2 AlignAttHooks는 결정2에 따라 폐기.** **사이드카 백엔드가 `StreamingAsrBackend`의 첫 구현**, 네이티브(mlx-rs/mlx-swift)가 drop-in.

### D.1 공통 타입

```rust
// crates/stt-core/src/asr/token.rs
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AsrToken {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub probability: Option<f32>,         // MLXWhisper는 None (조기확정 비활성, backends.py:209)
    pub detected_language: Option<String>,// ko/en 코드스위칭, 토큰/세그먼트 단위 (7장)
}
pub struct AsrResult { pub tokens: Vec<AsrToken>, pub raw_text: String, pub language: Option<String> }

#[derive(thiserror::Error, Debug)]
pub enum AsrError {
    #[error("backend not ready")] NotReady,
    #[error("sidecar crashed")] BackendDied,
    #[error("model not found offline: {0}")] ModelMissing(String),
    #[error("inference failed: {0}")] Inference(String),
    #[error("cancelled")] Cancelled,
}
```

### D.2 상위 trait — 파이프라인이 직접 호출 (async, B의 개선 채택)

사이드카 IPC가 본질적으로 async이므로 trait를 async로 승격(B). 네이티브 구현은 내부 동기 실행을 async로 감싸 drop-in 유지.

```rust
// crates/stt-core/src/asr/backend.rs
#[async_trait::async_trait]
pub trait StreamingAsrBackend: Send {
    async fn configure(&mut self, cfg: &AsrConfig) -> Result<(), AsrError>; // 모델경로/언어
    async fn warmup(&mut self) -> Result<(), AsrError>;                     // 번들 PCM 1회
    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64);
    async fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError>;
    fn get_buffer(&self) -> String;            // 미확정 partial (캐시값, 동기)
    fn start_silence(&mut self, at: f64);      // 침묵/리셋 훅
    fn on_change_speaker(&mut self, at: f64);  // 화자 전환 시 컨텍스트 리셋
    async fn finish(&mut self) -> Result<Vec<AsrToken>, AsrError>;
    fn set_language(&mut self, lang: Option<&str>); // None=auto(코드스위칭)
    fn caps(&self) -> BackendCaps;             // 런타임 능력 질의
}

/// 백엔드 이질성을 정책이 런타임에 분기(B의 베스트 아이디어).
pub struct BackendCaps {
    pub provides_word_timestamps: bool, // false면 LocalAgreement 불가 → segment-only 폴백
    pub provides_probability: bool,     // 조기확정(prob>0.95) 가능 여부
    pub self_streaming: bool,           // true=Voxtral/Qwen(자체 버퍼 재전사), false=Whisper류
    pub tokenizer_id: &'static str,     // hot-swap 시 토큰 체계 차이 식별 (D.6)
}
```

### D.3 하위 trait — LocalAgreement-2가 감싸는 "1회 추론" (Whisper류)

```rust
// backends.py:15 등가. OnlineAsrProcessor가 이 위에 올라가 StreamingAsrBackend를 구현.
pub trait WhisperLikeBackend: Send {
    fn transcribe(&self, audio: &[f32], init_prompt: &str) -> Result<AsrResult, AsrError>;
    fn ts_words(&self, r: &AsrResult) -> Vec<AsrToken>;   // 단어 타임스탬프 필수(전제)
    fn segments_end_ts(&self, r: &AsrResult) -> Vec<f64>; // segment 트리밍(한국어 강제)
    fn sep(&self) -> &str;                                // " "(en) vs ""(ko)
}
```

```rust
/// LocalAgreement-2 정책 래퍼(online_asr.py 1:1 이식). 그 자체로 StreamingAsrBackend.
pub struct OnlineAsrProcessor<B: WhisperLikeBackend> {
    backend: B,
    hyp: HypothesisBuffer,   // committed_in_buffer/buffer/new (online_asr.py:22-24)
    // 연속 2회 동일 최장공통접두사만 commit(:59-86), n-gram(1~5) 중복제거(:46-57),
    // segment 트리밍(한국어), 프롬프트 캐리오버 끝 200자(:187-209), UTF-8 pending(align_att_base.py:436-481)
}
#[async_trait::async_trait]
impl<B: WhisperLikeBackend + Send> StreamingAsrBackend for OnlineAsrProcessor<B> { /* ... */ }
```

### D.4 레지스트리 / 플랫폼 가용성 (B 채택)

```rust
pub enum BackendKind { SidecarMlxWhisper, MlxWhisper, Voxtral, QwenMetal } // qwen3-vllm(CUDA) 제외(결정6)
pub trait AsrBackendFactory: Send + Sync {
    fn create(&self, kind: BackendKind, model_id: &str) -> Result<Box<dyn StreamingAsrBackend>, AsrError>;
    fn available(&self) -> Vec<BackendKind>;  // iOS 시 QwenMetal/Sidecar* 제외(분석 8.2)
}
```

### D.5 결정2 "정책 단일화"의 정확한 의미 (세 심사 공통 모순 해소)

Voxtral/Qwen은 "매 iter 버퍼 전체 재전사"하는 **자체 스트리밍**(분석 2.5/2.6)이라 LocalAgreement-2를 쓰지 않는다. 이것이 결정2와 충돌하는가? — **충돌하지 않는다. 결정2의 적용 범위를 다음과 같이 확정한다:**

> **결정2 = "Whisper류(단어 타임스탬프 제공) 백엔드의 라이브 확정/partial 정책은 LocalAgreement-2로 단일화한다. AlignAtt/cross-attention 커스텀 디코더 경로는 일절 만들지 않는다."**

- `caps().self_streaming == false`(MLX Whisper) → 반드시 `OnlineAsrProcessor`(LocalAgreement-2)로 감싼다. 우리가 작성·유지하는 스트리밍 정책은 LocalAgreement **하나뿐**(AlignAtt 없음).
- `caps().self_streaming == true`(Voxtral/Qwen) → 백엔드 **내장** 정책을 그대로 사용(우리가 별도 정책을 추가하지 않음). 파이프라인은 `StreamingAsrBackend` 표면만 보므로 동일하게 다룬다.
- `caps().provides_word_timestamps == false`인 Whisper류가 등장하면 → segment-only 폴백(분석 2.3, 한국어 segment 강제와 정합). LocalAgreement는 segment end 타임스탬프 기준으로만 동작.

즉 "정책 단일화"는 **우리가 구현·유지하는 정책의 가짓수를 1개(LocalAgreement-2)로 고정**하고 AlignAtt를 배제한다는 뜻이며, 자체스트리밍 백엔드의 내장 정책은 정책 가짓수에 포함하지 않는다. (분석문서가 명시한 "세 정책 공존" 중 SimulStreaming/AlignAtt를 버리고, LocalAgreement + 백엔드내장 두 부류만 남긴다.)

### D.6 실행 중 백엔드 교체(hot-swap)의 전사 정합 (세 심사 공통 사각지대 해소)

seed AC는 "세 백엔드를 전환해 **비교**"를 요구한다. 새 백엔드는 토큰화/타임스탬프 체계가 다르다(`caps().tokenizer_id` 상이). 동일 세션 내 이미 확정된 전사와 어떻게 정합하는가:

- **확정(committed) 라인은 불변(immutable)으로 동결한다.** 전환은 "확정 경계"에서만 일어난다: `select_backend` → 현 백엔드 `finish()`로 진행 중 partial을 확정 → 그 시점까지를 동결 → 새 백엔드 `configure()`+`warmup()` → **새 백엔드는 동결 경계 이후의 오디오만 처리**.
- **전환 마커**를 transcript에 삽입(`TranscriptLine::kind = BackendSwitch{from,to,at}`)하여 UI가 "여기서부터 백엔드 B"임을 표시 → 비교가 명시적으로 가능(같은 세션, 다른 구간).
- 따라서 서로 다른 토큰 체계가 **한 라인 안에서 섞이지 않는다**. "세션 재시작"이 아니라 "동일 세션·확정경계 분할"이므로 자원 패널/메트릭/화자 갤러리는 연속 유지.
- **피크 메모리 가드**: Voxtral 4B(~3GB)+Qwen 1.7B(3.6GB) 동시 상주 시 OOM 위험(심사 지적). 전환은 **구 백엔드 `finish()`+드롭(모델 언로드) → 새 백엔드 로드** 순서로 직렬화하여 동시 상주를 피한다. 로딩 중 `BackendState::Switching` emit + 자원 패널이 언로드/로드 전이를 그대로 보여줌(ontology "관찰가능성"). 모델 디스크 예산은 G에서 관리.

### D.7 사이드카가 첫 구현 → 네이티브 drop-in

- **1단계**: `SidecarBackend: StreamingAsrBackend`. 내부적으로 C 프로토콜로 Python을 호출. LocalAgreement는 P0에서 사이드카 안(Python `online_asr.py` 재사용)에 둬 동작 우선 확보 → **[REFACTOR R-policy] P1에서 Rust `asr/policy`로 이전**(결정1의 2단계: 사이드카는 `WhisperLikeBackend.transcribe`만 수행하는 dumb transcriber로 축소, 정책은 Rust 소유). trait 경계 덕에 파이프라인 코드 변경 0.
- **2단계**: `MlxWhisperBackend: WhisperLikeBackend`(mlx-rs FFI) drop-in. **mlx-rs API 동등성**(rope/sdpa/conv1d/6bit quant) **[확인 필요, 오픈Q4]** = 2단계 진입 게이트.
- **3단계**: `MlxSwiftBackend`(`#[cfg(target_os="ios")]`, Rust↔Swift FFI) drop-in. 사이드카 crate는 링크 제외.

---

## E. Tauri IPC

### E.1 Rust→프론트 이벤트 (serde, `events.rs`)

모두 `#[derive(Serialize, Clone)]` + `#[serde(rename_all="camelCase")]`. 분석 2.7 `FrontData` 결함 교정: **타임스탬프 'H:MM:SS.cc' 문자열 → f64 초**, **speaker 매직넘버(-2/-1/양수, `timed_objects.py:162`) → enum**.

```rust
#[derive(serde::Serialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakerLabel {
    Silence,                              // 원본 -2
    Anonymous { id: u32 },                // Speaker N (세션 로컬)
    Enrolled  { id: String, name: String },// 등록 화자 (I)
}

#[derive(serde::Serialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LineKind {
    Speech {
        speaker: SpeakerLabel,
        committed: bool,                  // true=확정, false=partial
        // 한·영 intra-sentence 코드스위칭: 토큰 span별 언어를 보존(7장 해소)
        spans: Vec<LangSpan>,             // [{ text, lang: Option<"ko"|"en"> }]
    },
    BackendSwitch { from: String, to: String, at: f64 }, // hot-swap 마커 (D.6)
}

#[derive(serde::Serialize, Clone)]
pub struct LangSpan { pub text: String, pub lang: Option<String> }

#[derive(serde::Serialize, Clone)]
pub struct TranscriptLine {
    pub id: u64,
    pub start: f64, pub end: f64,
    pub line: LineKind,
}

/// transcript_update : 누적이 아니라 diff. relabeled는 과거 라인 speaker 소급 정정(I.1).
#[derive(serde::Serialize, Clone)]
pub struct TranscriptSnapshot {
    pub added: Vec<TranscriptLine>,
    pub updated: Vec<TranscriptLine>,                 // 같은 id 교체
    pub relabeled: Vec<SpeakerRelabel>,               // {line_id, new_speaker} 소급 정정
    pub buffer_text: String,                          // partial
    pub buffer_speaker: SpeakerLabel,
    pub remaining_audio_sec: f64,                      // 처리 지연(lag) → UI 경고
}
#[derive(serde::Serialize, Clone)]
pub struct SpeakerRelabel { pub line_id: u64, pub new_speaker: SpeakerLabel }

/// metrics_update (1초)
#[derive(serde::Serialize, Clone)]
pub struct MetricsSnapshot {
    pub cpu_pct: f32,
    pub rss_mb: f32,            // 앱 + 사이드카 RSS 합산
    pub sidecar_rss_mb: Option<f32>,
    pub metal_mb: Option<f32>, // GPU/Metal 힙 추정(불가시 None, 분석 6.3 [확인 필요])
    pub rtf: f32,
    pub latency_ms_p50: f32, pub latency_ms_p95: f32,
    pub queue_depth: u32,
    pub backend: String, pub model: String,
}

// transcript_done { sessionId, lineCount, durationSec, savedPath? }
// model_download_progress { modelId, receivedBytes, totalBytes, phase: "download"|"verify"|"done" }
// backend_state { kind, modelId, status: "loading"|"ready"|"switching"|"error" }
// capture_state { source, state: "active"|"overrun"|"permission_denied" }
// error { kind: "permission"|"model_missing"|"sidecar_crash"|"input_overrun"|"thermal", message, recoverable }
```

이벤트 채널: `transcript_update` / `metrics_update` / `transcript_done` / `model_download_progress` / `backend_state` / `capture_state` / `error`.

### E.2 프론트→Rust command (`commands.rs`)

```rust
#[tauri::command] async fn start_session(input: InputSelection, backend: BackendKind, model_id: String, lang: Option<String>) -> Result<SessionId, String>;
#[tauri::command] async fn stop_session(id: SessionId) -> Result<(), String>;           // CancellationToken trip
#[tauri::command] async fn select_backend(kind: BackendKind, model_id: String) -> Result<(), String>; // hot-swap (D.6)
#[tauri::command] async fn list_inputs() -> Result<Vec<AudioInput>, String>;
#[tauri::command] async fn select_input(sel: InputSelection) -> Result<(), String>;     // InputSelection{ mic, system } — iOS는 system=false 강제
#[tauri::command] async fn list_models() -> Result<Vec<ModelCatalogEntry>, String>;     // 카탈로그 + 로컬 보유/용량
#[tauri::command] async fn download_model(model_id: String) -> Result<(), String>;      // 진행은 이벤트
#[tauri::command] async fn delete_model(model_id: String) -> Result<(), String>;        // 디스크 예산 관리(G)
#[tauri::command] async fn export_transcript(id: SessionId, format: ExportFormat, path: String) -> Result<String, String>;
#[tauri::command] async fn enroll_speaker(name: String, sample: EnrollRef) -> Result<String, String>; // I.2
#[tauri::command] async fn list_speakers() -> Result<Vec<SpeakerProfile>, String>;
#[tauri::command] async fn get_config() -> Result<AppConfig, String>;
#[tauri::command] async fn set_config(cfg: AppConfig) -> Result<(), String>;
```

기존 `greet`는 제거(또는 헬스체크 `ping`으로 대체).

### E.3 capabilities 권한 (데스크톱/모바일 분리)

`src-tauri/capabilities/default.json`(기존 `core:default`,`opener:default`)에 추가, 그리고 **`mobile.json` 신규 분리**:
- `core:event:default`(emit/listen).
- **shell sidecar scope 최소화**: `shell:default` 대신 명시적 sidecar 권한 객체로 `name: "stt-mlx-sidecar"` **단 하나만 허용**(임의 명령 실행 차단 = 클라우드 0/보안 추가 방어선, C의 베스트 아이디어).
- `dialog:allow-save` + `fs:allow-write-file`(scope = app_data_dir + 사용자 선택 export 경로).
- `store:default`(config·갤러리 경량 저장, `tauri-plugin-store`).
- **`capabilities/mobile.json`**(`platforms:["iOS"]`)은 **shell sidecar 권한을 제외**(iOS 사이드카 불가, 결정3). 데스크톱/모바일 capability 분리가 Tauri 2 관용.

---

## F. 프론트 (React 19 + TS + Vite)

기존 `src/App.tsx` greet 예제를 전면 교체(라우팅 없는 단일 화면 + 패널).

```
src/
├── main.tsx                 # 유지
├── App.tsx                  # 레이아웃: 좌(전사 뷰) / 우(컨트롤·모니터)
├── lib/
│   ├── ipc.ts               # invoke 래퍼 + listen 등록
│   └── events.ts            # Rust serde 페이로드와 1:1 TS 타입 (P1에 ts-rs 자동생성 검토 [REFACTOR R-tsrs])
├── hooks/
│   ├── useTranscript.ts     # transcript_update 구독 → added/updated/relabeled 머지
│   ├── useMetrics.ts        # metrics_update 구독(1초)
│   └── useModelDownload.ts  # model_download_progress 구독
└── components/
    ├── TranscriptView.tsx   # 화자 라벨 + 확정(불투명)/partial(회색 italic) + BackendSwitch 구분선
    ├── LangSpanText.tsx     # span별 ko/en 뱃지(intra-sentence 혼용 렌더, 7장)
    ├── SpeakerBadge.tsx     # Anonymous(색 순환)/Enrolled(이름) + 등록 버튼, relabel 시 색·이름 갱신
    ├── ResourceMonitor.tsx  # CPU/RAM(앱+사이드카)/RTF/지연 p50·p95 게이지 + sparkline (1초)
    ├── BackendSelector.tsx  # MLX Whisper/Voxtral/qwen 라디오 (iOS는 가용분만), Switching 스피너
    ├── InputSelector.tsx    # mic ☑ / system ☑ (iOS는 system 숨김)
    ├── ModelManagerPanel.tsx# 카탈로그 + 다운로드/삭제/진행바/SHA/디스크 사용량
    └── ExportBar.tsx        # txt/srt/json → export_transcript → 저장 dialog
```

핵심 UX:
- **확정/partial 시각 구분**(seed AC): 확정 라인 고정, `buffer_text`는 마지막 줄 회색 italic.
- **화자 구분 + 소급 정정**(seed AC, I.1): `SpeakerLabel` 색상. `relabeled` 수신 시 해당 line_id의 색·이름을 소급 갱신(익명→등록 통합).
- **한·영 혼용**: `LangSpan[]`을 span 단위 ko/en 뱃지로 렌더(7장).
- **백엔드 전환**(ontology "교체가능성"): `BackendSwitch` 마커를 구분선으로, 전환 중 Switching 스피너.
- **자원 패널 상시 노출**(ontology "관찰가능성"): 1초 갱신. RTF>1 경고색.

---

## G. 모델 매니저 (클라우드 0 보장)

`crates/stt-core/src/models/`. 분석 6.1/6.2 + 오픈Q7.

- **카탈로그**: `src-tauri/resources/model_catalog.json`(원본 `MLX_MODEL_MAPPING`/`cli.py:142-187` 등가). 엔트리 = `{id, backend, repo, revision(commit SHA), files:[{path, sha256, size}], size_mb, languages, platforms}`. **revision=commit SHA 핀**(원본 latest=재현성 위험, 6.2).
- **다운로드**: `hf-hub` crate 또는 `reqwest`로 HF resolve API 직접. `model_download_progress` 스트리밍. **`download_model` 명시 호출에서만** 발생. **resume vs 폐기**: 부분 파일은 `.partial` suffix로 받고 완결+SHA 검증 후 rename, 앱 종료 시 `.partial`은 재개 대상(없으면 폐기) — 심사 지적 해소.
- **캐시 단일화**: `~/.cache/huggingface` 의존 제거 → `app_data_dir/models/<id>/<sha>/`(원본 `model_cache_dir` 무시 버그 정리, `backends.py:163`).
- **무결성**: 파일별 SHA256(`sha2`) 검증(원본 OpenAI 경로 SHA를 전 모델 확대, 6.2). 불일치 시 폐기·재시도.
- **디스크 예산/축출**(심사 지적 해소): 결정6 3종 동시 보유 시 수 GB(Voxtral 3GB+Qwen 3.6GB+Whisper). `cache.rs`가 총 사용량을 추적해 `list_models`에 노출, LRU 축출 후보 표시 + `delete_model`. iOS는 소형 모델만 카탈로그 노출(8.2).
- **warmup**: github JFK wav(`warmup.py:19`) 대신 **번들 `warmup_16k.f32`**. **[확인 필요]** 모델별 기대 입력 길이/형식 차이(Whisper 30s 패딩, Voxtral 80ms/token=1280샘플 `spectrogram.py:21`, Qwen 버퍼 재전사) → warmup 샘플을 **30초 길이 단일 PCM**으로 만들어 3종을 모두 데우되, 각 사이드카가 자기 형식으로 잘라 쓰도록 함(심사 지적 해소).

### G.1 클라우드 0 코드 보장 지점

1. **다운로드/추론 단계 물리 분리**: `models::downloader`(reqwest/hf-hub, `desktop` feature)와 추론(`asr`/`vad`/`diar`)은 다른 모듈·feature. 추론에 네트워크 클라이언트 미주입.
2. **추론 경로는 로컬 절대경로만**: 미존재 시 네트워크 폴백 없이 `AsrError::ModelMissing` 즉시 반환(원본 자동 다운로드, `model_paths.py:202` 역전).
3. **사이드카 환경 격리**: spawn 시 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` 주입(원본 grep 0건, 6.1).
4. **백엔드 화이트리스트**: OpenAI API 백엔드(`backends.py:235`)·qwen3-vllm(CUDA)은 enum/팩토리에서 제외(결정6).
5. **CI 불변식**(B): 추론 crate 의존 그래프에 HTTP 클라이언트 부재를 grep 단언(회귀 방지) + capabilities의 임의 shell 차단(E.3).
6. **검증**: AC 1번("인터넷 차단 상태 동작")을 **P0 exit 게이트로 비행기모드 통합테스트**(A).

---

## H. 자원 모니터

`crates/stt-core/src/metrics/` + 독립 tokio interval 태스크. 분석 6.3(라이브 소스 없음 → 신규)/오픈Q9.

- **CPU/RAM**: `sysinfo`(또는 mach `task_info` 정밀 RSS). **사이드카 자식 프로세스 RSS를 PID로 합산**(`sidecar_rss_mb` 별도 표기).
- **RTF/지연**: `SessionMetrics`(rtf, p50/p95 latency, queue depth, n_tokens) 이식(`metrics_collector.py`). RTF = 추론시간/오디오길이. 지연 = 발화 end → 확정 emit.
- **emit**: 1초 interval → `metrics_update`. State actor에서 queue depth/rtf, system.rs에서 CPU/RAM.
- **[확인 필요] Metal/GPU 힙**: 심사 지적 — 추론 메모리 대부분이 사이드카의 MLX/Metal 측(GPU 힙)이라 RSS만으론 실제 점유를 못 잡는다. `metal_mb`는 가능하면 Metal API(`MTLDevice.currentAllocatedSize`, Mac 네이티브 단계) 또는 사이드카가 `mx.metal.get_active_memory()` 등으로 보고하는 값을 RESULT 프레임에 실어 노출, 불가 시 `None`. ANE/GPU 사용률(%)은 IOReport/powermetrics 권한 이슈로 1차 제외. UI에 "표시 RAM ≠ GPU 점유" 주석 노출.

---

## I. 화자분리 / 등록·식별 (신규 설계)

분석 2.2(Sortformer: PyTorch/450MB/4화자/MPS 미사용 `sortformer_backend.py:62` → 온디바이스 부적합) + 오픈Q5·Q6. `crates/stt-core/src/diar/`.

### I.1 온라인 화자분리 + 청크 간 동일성 유지 + reconciliation (세 심사 공통 사각지대 정면 해소)

- **모델 = `ort`(ONNX)**: PyTorch/NeMo 직접 이식 불가 → onnx 단일화. 후보: sherpa-onnx 계열(segmentation + speaker embedding) 또는 pyannote-onnx. **[확인 필요, 오픈Q6]** 한·영·4화자 초과·동시발화 벤치.
- **청크 간 화자 동일성**(`tracker.rs`): Sortformer의 `spkcache`/`fifo`(`:142-158`)를 직접 못 옮기므로, **온라인 임베딩 클러스터링 + 활성 화자 트랙(track)** 으로 대체한다. 각 청크의 화자 세그먼트 임베딩을 기존 트랙의 centroid와 cosine 매칭, 임계값 초과면 동일 트랙(같은 `Anonymous{id}`) 유지·centroid 갱신, 미만이면 신규 트랙. **이것이 "Speaker 1이 청크 경계를 넘어 일관 유지"의 핵심 메커니즘**(심사가 "모델만 교체로 과소평가"라 지적한 지점).
- **toᴋen⨉화자 정렬**: `alignment.rs`가 `tokens_alignment.get_lines_diarization()`(`:182-196`, intersection_duration 최대 화자) 이식. B.5 클램핑과 동일 모듈.
- **reconciliation(소급 정정)**: 라이브 트랙은 나중 정보로 합쳐질 수 있다(초기 Speaker 2가 실은 Speaker 1). 또는 등록 화자로 식별이 늦게 확정된다. 이때 **이미 emit된 과거 라인의 speaker를 소급 변경**해야 한다 → E.1 `TranscriptSnapshot.relabeled: Vec<SpeakerRelabel>` 로 표현(diff/coalesce 모델에서 유실 방지, B.4 ③). **seed C5("실시간 라이브 전용, 사후 일괄 재정리 없음")와의 긴장 해소**: 이 relabel은 회의 종료 후 일괄 재처리가 아니라 **라이브 진행 중 점진적 정정**(화자 트랙이 더 많은 발화를 본 결과)이므로 C5에 부합한다. 종료 후 추가 재정리는 하지 않는다.

```rust
#[async_trait::async_trait]
pub trait OnlineDiarBackend: Send {
    async fn accept_pcm(&mut self, pcm: &[f32], t_start: f64, t_end: f64)
        -> Result<DiarUpdate, DiarError>;
    fn reset(&mut self);
}
pub struct SpeakerSegment { pub start: f64, pub end: f64, pub track_id: u32, pub embedding: Option<Vec<f32>> }
pub struct DiarUpdate { pub segments: Vec<SpeakerSegment>, pub relabels: Vec<(u32 /*old track*/, u32 /*merged into*/)> }
```

### I.2 등록 / 식별 (참고 코드 0건, 전체 신규)

- **임베딩**: ECAPA-TDNN 또는 wespeaker(onnx, `ort`) → 화자 d-vector. 온라인 diar의 embedding 모델과 공유 가능.
- **갤러리 DB**: `app_data_dir/speakers.db`(SQLite via `rusqlite` 또는 `tauri-plugin-store`). 스키마 `{id, name, embeddings:[Vec<f32>](다중 발화 centroid), enrolled_at}`.
- **등록**: `enroll_speaker(name, sample)` → 수 초 발화 임베딩 평균 저장. 라이브 중 익명 트랙을 사후 명명도 지원(→ I.1 relabel로 과거 라인 갱신).
- **식별**: 트랙 임베딩 vs 갤러리 cosine. 임계 초과 → `Enrolled{name}`, 미만 → `Anonymous{id}`.
- **[확인 필요]** 임계값/등록 길이/트랙↔등록 라벨 안정성은 실측 튜닝. P0는 diar OFF(단일 화자 가정)로 격리 **[REFACTOR R-diar]**, P1.5에서 본격화.

---

## J. 단계별 로드맵 (작업마다 커밋, 각 단계 = 동작하는 산출물)

> 커밋 메타에 AI 도구 표기 금지(seed C8). 미지수 제거 순서(A의 리스크 번다운): 사이드카 프로토콜 → MLX 단어 ts → LocalAgreement 동작 → 오프라인 → 정합/화자.

### P0 — 스캐폴드 정비 (반나절)
**산출물**: 기존 `pnpm tauri dev` / `pnpm tauri ios build`가 깨지지 않은 채 골격만 추가.
- 커밋1: `src-tauri/Cargo.toml` 워크스페이스화 + 빈 `stt-core`/`stt-sidecar-proto` 추가, `pnpm tauri dev`·iOS 빌드 무결 확인.
- 커밋2: `tauri-plugin-shell`/`tauri-plugin-store`/`tauri-plugin-dialog` 의존·init·capabilities(default+mobile) 추가.
- 커밋3: greet 제거, `commands.rs`/`events.rs`/`app_state.rs` 빈 스텁 + 프론트 빈 레이아웃 셸.

### P1 — 사이드카 수직 슬라이스 PoC (Mac, 최우선 리스크 제거)
**산출물**: Mac 마이크 → 화면 라이브 전사 끝까지 동작 + 오프라인.
- 커밋4: cpal 마이크 → rubato 16k mono f32 → `mpsc`, 콘솔 RMS 확인.
- 커밋5: 사이드카 부트 + UDS PCM + stdout NDJSON `ready` 핸드셰이크 echo. **미지수#1(프로토콜) 제거.** (PyInstaller+MLX/Metal dylib·셰이더 번들이 실제 Metal 추론하는지 + **ad-hoc/hardened runtime 서명·공증 통과**까지 이 커밋에서 검증 — 심사 지적 배포 리스크.)
- 커밋6: 사이드카에 MLX Whisper `transcribe(word_timestamps=True)` 연결, **단어 ts 제공 검증**. **미지수#2 제거.**
- 커밋7: 사이드카 내부 LocalAgreement(Python `online_asr` 재사용, P0a) → committed/buffer 분리. **미지수#3 제거.**
- 커밋8: `SidecarBackend`(StreamingAsrBackend 첫 구현) + 생명주기/크래시 복구(60s/3회 가드).
- 커밋9: 최소 파이프라인(capture→사이드카→state actor→`transcript_update`) + ts pass-through(드리프트 격리, B.5) + 프론트 TranscriptView(확정/partial) + ControlBar.
- 커밋10: 모델 매니저(카탈로그/SHA핀/캐시/오프라인 강제) + ModelManagerPanel + `model_download_progress`.
- 커밋11: 오프라인 강제(HF_HUB_OFFLINE) + **비행기모드 통합테스트(AC1 게이트)**.
- 커밋12: metrics(CPU/RAM/RTF/지연, 사이드카 RSS 합산) + `metrics_update` + ResourceMonitor.
- 커밋13: export(txt/srt/json) + `transcript_done` + ExportBar.

### P1.5 — 화자분리 + 시스템오디오 + 등록/식별 (Mac)
**산출물**: mic+system 동시 + 화자 라벨(익명) + 등록/식별 + AC "2명 구분".
- 커밋14: ScreenCaptureKit 시스템오디오(화면녹화 권한) + `mixer.rs` 2소스 정렬/믹싱(B.6).
- 커밋15: diar onnx(sherpa/pyannote) + `tracker.rs` 청크 간 동일성(I.1).
- 커밋16: `alignment.rs` 토큰⨉화자 정렬 + reconciliation/relabel IPC(I.1) → UI 소급 갱신.
- 커밋17: 임베딩+갤러리 등록/식별(I.2) + enroll/list command.

### P2 — Rust 파이프라인 굳히기 + 네이티브 MLX (Mac)
**산출물**: 정책/VAD/정렬 전부 Rust, 텐서만 MLX. 백엔드 hot-swap·3종 동작.
- 커밋18: VAD(Silero ort, 512프레임 히스테리시스) Rust 이식(오픈Q11 노드명 검증) → PcmEvent fan-out. **골든 회귀: 원본 ort 출력과 수치 비교(J.1).**
- 커밋19: LocalAgreement Rust 이전(R-policy, 결정1 2단계) → 사이드카 dumb transcriber 축소. **골든 회귀: Python vs Rust committed/buffer 토큰 시퀀스 바이트 일치(J.1).**
- 커밋20: State actor + bounded mpsc 백프레셔 + 타임스탬프 클램핑 정밀 이식(오픈Q8).
- 커밋21: `MlxWhisperBackend`(mlx-rs) drop-in + mlx-rs 동등성 검증(오픈Q4).
- 커밋22: Voxtral(MLX) 자체스트리밍 백엔드(D.5) + tekken.json 파서.
- 커밋23: qwen Metal 백엔드(Mac 전용) + LIS 보간/CJK 분할 이식.
- 커밋24: `select_backend` hot-swap(D.6, 확정경계 분할 + 피크 메모리 가드) + intra-sentence 코드스위칭 렌더(7장).
- 커밋25: `stt-core` 헤드리스 통합테스트 스위트 정비(J.1).

### P3 — iOS 마이크 전용 (후순위, 결정3)
**산출물**: iPhone 마이크 라이브 전사(시스템/통화오디오 범위 밖).
- 커밋26: AVAudioEngine 캡처(Swift FFI, `gen/apple/Sources/app` 활용) + `NSMicrophoneUsageDescription`.
- 커밋27: `MlxSwiftBackend`(Rust↔Swift FFI) drop-in + 소형 모델 카탈로그(Whisper tiny/small, Qwen 0.6B).
- 커밋28: mobile capability·thermal/배터리 가드(seed C12) + On-Demand Resources 검토.
- 커밋29(조사 트랙, 비차단): **iOS 시스템/통화 오디오 "모색" PoC**(ReplayKit/Broadcast Upload Extension 한계 문서화) — seed C7 의무 이행, 결정3(범위 밖)과의 긴장을 조사 산출물로 남김(세 심사 공통 지적). 제품 기능 아닌 조사 노트.

### J.1 정합성 검증 전략 (세 심사 공통 사각지대 — 신규)
seed AC "VAD·화자분리가 whisper-live-kit와 동등 수준 재현"을 측정 가능하게:
- **골든 트랜스크립트 회귀**: 고정 입력 WAV(한·영 샘플) → `stt-core` 헤드리스 실행 → 기대 `PipelineEvent` 시퀀스/committed 토큰을 스냅샷. P1(Python 사이드카)에서 기준선 캡처 → P2(Rust 정책 이전) 시 **바이트/타임스탬프 허용오차 내 일치**를 CI가 단언(R-policy 회귀 방지).
- **VAD 수치 단위테스트**: 512프레임 입력에 대한 ort `(out_prob, new_state)`를 원본 silero와 비교(오픈Q11 노드명 확정 포함).
- **acceptance 메트릭 임계 정의**: "저지연 수 초" = 발화 end→확정 emit p95 < 3s(P1 측정). "화자 2명 구분" = 2화자 고정 클립에서 트랙 분리 정확도 측정.

---

## K. 기존 `app/` 스캐폴드 매핑 (구체 경로)

| 작업 | 경로 | 변경 |
|---|---|---|
| 워크스페이스 선언 + 어댑터 deps | `src-tauri/Cargo.toml` | `[workspace] members=[".","crates/*"]`. deps에 `tokio`(rt-multi-thread,sync,process,macros), `tokio-util`, `tauri-plugin-shell`/`-store`/`-dialog`, `thiserror`, `async-trait`. `serde`/`serde_json` 유지. `[lib] name=app_lib`/crate-type 유지. |
| 코어 crate 신설 | `src-tauri/crates/stt-core/**` | 신규(A.1). deps: `serde`,`tokio`,`async-trait`,`thiserror`,`ort`(opt),`rubato`,`sha2`,`sysinfo`(opt),`rusqlite`(opt),`hf-hub`/`reqwest`(desktop feat). tauri 무의존. |
| proto crate | `src-tauri/crates/stt-sidecar-proto/**` | 신규(C.2 serde 스키마) |
| 사이드카 클라이언트 | `src-tauri/crates/stt-asr-sidecar/**` | 신규, `#[cfg(all(macos,feature="sidecar"))]` StreamingAsrBackend 첫 구현 |
| 캡처(플랫폼) | `src-tauri/src/capture/**` | 신규. cpal(`[target.'cfg(target_os="macos")'].dependencies` objc2-screen-capture-kit), AVAudioEngine FFI(P3). mixer/resample. |
| Tauri 어댑터 | `src-tauri/src/lib.rs` | greet 제거. `.plugin(shell::init())`,`.plugin(store::init())`,`.plugin(dialog::init())`, `.manage(AppState)`, `.invoke_handler(generate_handler![start_session, stop_session, select_backend, select_input, list_inputs, list_models, download_model, delete_model, export_transcript, enroll_speaker, list_speakers, get_config, set_config])`, core `broadcast<PipelineEvent>`↔emit 브리지. |
| main.rs | `src-tauri/src/main.rs` | 변경 없음(`app_lib::run()`) |
| 커맨드/이벤트/상태 | `src-tauri/src/{commands.rs,events.rs,app_state.rs}` | 신규(E) |
| 데스크톱 권한 | `src-tauri/capabilities/default.json` | event/shell(sidecar scope=stt-mlx-sidecar 1개)/fs/dialog/store 추가(E.3) |
| 모바일 권한 | `src-tauri/capabilities/mobile.json` | 신규(iOS, shell 제외) |
| 사이드카 번들 | `src-tauri/tauri.conf.json` | `bundle.externalBin:["binaries/stt-mlx-sidecar"]`, `bundle.resources:["resources/model_catalog.json","resources/warmup_16k.f32"]`. csp=null 유지(로컬 전용). `beforeDevCommand`에 사이드카 빌드 스텝 검토. iOS 빌드에서 externalBin 제외. |
| 사이드카 소스 | `src-tauri/sidecar/stt_mlx/**` + PyInstaller spec | 신규(C.3) |
| 사이드카 바이너리 | `src-tauri/binaries/stt-mlx-sidecar-<triple>` | 빌드 산출물(gitignore 가능) |
| 카탈로그·warmup | `src-tauri/resources/{model_catalog.json,warmup_16k.f32}` | 신규(G) |
| 프론트 진입 | `src/App.tsx` | greet 예제 → F 레이아웃 셸 교체 |
| 프론트 IPC/훅/컴포넌트 | `src/{lib,hooks,components}/**` | 신규(F) |
| 프론트 deps | `package.json` | `@tauri-apps/plugin-shell`,`@tauri-apps/plugin-store`,`@tauri-apps/plugin-dialog` 추가. 차트는 경량 자체구현 권장. |
| macOS 권한 | macOS 타깃 Info.plist | `NSMicrophoneUsageDescription` + 화면녹화(ScreenCaptureKit) (P1.5) |
| iOS 마이크 권한 | `src-tauri/gen/apple/app_iOS/Info.plist` | `NSMicrophoneUsageDescription` (P3) |
| iOS 엔타이틀먼트 | `src-tauri/gen/apple/app_iOS/app_iOS.entitlements` | P3 필요 시 |
| Swift FFI | `src-tauri/gen/apple/Sources/app/` | P3: AVAudioEngine·mlx-swift 브리지(`main.mm`/`bindings` 옆). **검증됨**: `gen/apple/project.yml`의 `Build Rust Code` preBuildScript + `Externals/{arm64,x86_64}/${CONFIGURATION}/libapp.a` 링크 + `UIRequiredDeviceCapabilities:[arm64,metal]`가 이미 갖춰져 iOS Rust 통합 추가설정 최소. |
| Android gen | `src-tauri/gen/android/**` | 1차 범위 밖(seed). 건드리지 않음. |

---

## 핵심 리스크 / [확인 필요] 요약 (오픈 퀘스천 연계)

1. **[최상위]** 사이드카 PyInstaller+MLX/Metal 번들 동작 + 코드서명·공증(hardened runtime) 통과 — P1 커밋5 최초 검증(심사 지적, 배포 자체 차단 리스크).
2. **[최상위]** 타임스탬프 정합/TokensAlignment 이식(오픈Q8) — P0 pass-through 격리, P2 커밋20 정밀 이식. 긴 회의 드리프트.
3. **[高]** MLX Whisper 단어 타임스탬프 제공(LocalAgreement 전제) — P1 커밋6 검증. 미제공 시 segment-only 폴백(D.5).
4. **[高]** 온라인 diar 청크 간 화자 동일성 + 등록/식별 reconciliation(오픈Q5·Q6) — I.1/I.2, P1.5. relabel은 라이브 점진 정정으로 seed C5와 정합(I.1).
5. **[中]** mlx-rs API 동등성(rope/sdpa/conv1d/6bit, 오픈Q4) — P2 커밋21 게이트.
6. **[中]** 결정2 vs 자체스트리밍 모순 — D.5에서 "정책 가짓수 1개(LocalAgreement)+AlignAtt 배제"로 확정 해소.
7. **[中]** hot-swap 전사 정합/피크 메모리 OOM(seed AC) — D.6 확정경계 분할 + 직렬 언로드/로드.
8. **[中]** mic+system 클록 드리프트/에코(seed C7) — B.6 공통 클록 리스탬프+단일 mono 믹스, 에코 억제 [확인 필요].
9. **[中]** 한·영 intra-sentence 코드스위칭 표시(7장) — `LangSpan[]` span 단위 언어 보존+UI 뱃지로 해소.
10. **[低]** Silero ONNX 노드명(오픈Q11), 모델 실측 크기/RAM(오픈Q12), Metal/GPU 메모리 관측(오픈Q9, H), 모델별 코드스위칭 정확도(오픈Q10), 디스크 예산/축출(G) — PoC 중 실측.
11. **[조사 트랙]** iOS 시스템/통화 오디오(오픈Q1) — 결정3으로 범위 밖이나 seed C7 "모색" 의무를 P3 커밋29 조사 노트로 이행.

> 본 확정안은 분석문서(`01-whisper-live-kit-analysis.md`)와 seed(`00-product-seed.md`) 및 확정결정 6항에 정합한다. 관련 스캐폴드: `app/src-tauri/{Cargo.toml,src/lib.rs,tauri.conf.json,capabilities/default.json}`, `app/src-tauri/gen/apple/project.yml`, `app/{package.json,src/App.tsx}`.
