# whisper-live-kit 분석 — 온디바이스 Tauri 회의 전사 앱 포팅 근거

> 목적: 온디바이스(클라우드 0) 회의 전사 앱(Tauri + Rust 백엔드, Mac → iOS 우선)을 만들 때, whisper-live-kit의 각 서브시스템을 (a)Rust로 포팅 / (b)Apple 네이티브 API로 대체 / (c)알고리즘만 참고 / (d)MLX로 모델만 실행 중 무엇으로 처리할지 판단할 근거를 제공한다.
>
> 표기 규칙: 코드 직접 확인된 사실은 `파일:라인`으로 근거를 단다. 코드/모델 가중치를 열지 못해 추정인 항목은 **확인 필요**로 명시한다.

---

## 1. 개요 — 전체 아키텍처와 라이브 전사 파이프라인

whisper-live-kit은 **세션(=WebSocket 연결)당 비동기 producer-consumer 파이프라인**이다. 인코딩된 바이트(또는 raw PCM)를 받아 → 음성/침묵 분리(Silero VAC) → 활성 음성만 ASR/화자분리 큐로 fan-out → 결과를 공유 `State`에 병합 → `FrontData` 스냅샷을 클라이언트로 스트리밍한다. 오케스트레이션은 전부 `asyncio` 기반이고, 무거운 추론은 `asyncio.to_thread`로 이벤트 루프 밖으로 오프로드한다(`audio_processor.py:408`).

```
[브라우저 캡처]  마이크(getUserMedia) / 탭오디오(chrome.tabCapture)
      │  AudioWorklet(pcm_worklet.js) → recorder_worker.js (48k→16k, s16le)  또는  MediaRecorder(WebM)
      ▼  WebSocket binary
┌─────────────────────────────────────────────────────────────────────────────┐
│ AudioProcessor (세션당 1개, audio_processor.py:54)                              │
│                                                                               │
│  process_audio() ─encoded─▶ FFmpegManager(subprocess) ─s16le 16k mono─┐       │
│       (audio_processor.py:709)   (ffmpeg_manager.py:60)                │       │
│                  └──pcm_input=True (raw PCM 직행)──────────────────────┤       │
│                                                                        ▼       │
│  handle_pcm_data (audio_processor.py:753): int16→f32, Silero VAC,             │
│    start/end 이벤트로 [active][silence][active] 슬라이스 (audio_processor:790) │
│                          │                                                     │
│         ┌────────────────┼─────────────────────┐                              │
│         ▼                ▼                     ▼                              │
│  transcription_queue   diarization_queue   (translation은 토큰 구동)            │
│         │                │                                                     │
│         ▼ to_thread      ▼ await                                               │
│  transcription_processor  diarization_processor   translation_processor        │
│  (process_iter)          (sortformer/diart)       (nllb)                       │
│   online_asr / simul     SpeakerSegment 누적                                    │
│         │                │                                                     │
│         └────────┬───────┘                                                     │
│                  ▼  공유 State (asyncio.Lock, timed_objects.py:217)            │
│  results_formatter (50ms 폴링): TokensAlignment.update()+get_lines()           │
│    = ASR 토큰 ⨉ 화자구간 ⨉ 침묵 정렬 → FrontData (audio_processor.py:552)      │
└──────────────────────────────────┬────────────────────────────────────────────┘
                                    ▼  WebSocket JSON (full 또는 diff)
                          [브라우저 렌더] live_transcription.js
```

핵심 설계 특성:

- **VAC가 추론 비용의 게이트**: 활성 음성만 큐로 들어가고 침묵은 타임스탬프만 누적된다(`audio_processor.py:188`, `_enqueue_active_audio`).
- **세 가지 라이브 정책이 공존**: LocalAgreement-2(`local_agreement/online_asr.py`), SimulStreaming/AlignAtt(`simul_whisper/`), 그리고 백엔드 자체 스트리밍(Voxtral/Qwen은 buffer 재전사 방식).
- **공유 모델 + 세션별 상태**: Silero 세션은 공유 stateless `OnnxSession`, 상태는 세션별 `OnnxWrapper`(`core.py:104`, `audio_processor.py:96`). ASR은 전역 `threading.Lock`으로 직렬화(`thread_safety.py:34`).
- **단일 사용자 온디바이스 앱에서는 과설계인 부분**: 전역 모델 락, FFmpeg 상태머신/재시작, WebSocket 다중 세션 — 모두 단순화/제거 대상.

---

## 2. 서브시스템별 분석 (핵심 메커니즘 + file:line 근거)

### 2.1 VAD (Silero VAC) — 파이프라인 게이트
- **두 개의 다른 플래그**: `vac`(Silero 외부 게이트, `config.py:34` 기본 True)와 `vad`(ASR 백엔드 내부 no_speech_prob 필터, `config.py:40`, `local_agreement/backends.py:245`)는 독립이다. 혼동 금지.
- **엄격한 프레임 계약**: 16kHz에서 **정확히 512 샘플**(8kHz는 256). 위반 시 예외(`silero_vad_iterator.py:95-96`, 확인됨). `FixedVADIterator`가 임의 길이 입력을 512 프레임으로 버퍼링(`silero_vad_iterator.py:312-314`). 512 샘플 @16k = 32ms hop.
- **재귀 상태 모델**: `_state` shape `(2,1,128)` + 프레임 앞에 붙는 64샘플 `_context`(`silero_vad_iterator.py:85,99`, 확인됨). sr/batch 변경 시 reset(`:101-106`). 순수 per-frame 분류기가 **아님** — 상태 보존 필수.
- **히스테리시스 상태머신**: prob≥0.5면 SPEECH 트리거(`{'start'}` 백데이트), triggered 중 prob<0.35(=threshold-0.15)면 temp_end 타이머 시작, `min_silence_samples` 경과 후에만 `{'end'}` emit(`silero_vad_iterator.py:266-285`). 기본 threshold=0.5, min_silence=100ms, speech_pad=30ms.
- **ONNX IO**: 입력 `input`/`state`/`sr`, 출력 `(out_prob, new_state)` — **호출 코드에서 추론한 것이며 .onnx 그래프를 직접 열어 노드명을 확인하지 못함(확인 필요, `silero_vad_iterator.py:113-116`)**.
- **모델 자산**(실측): `silero_vad.onnx` 2,327,524B(기본), `silero_vad_16k_op15.onnx` 1,289,603B, `silero_vad_half.onnx` 1,280,395B(fp16, 로더 미참조), `silero_vad.jit` 2,271,162B. 디렉터리 직접 확인됨.

### 2.2 Diarization (화자분리) — 두 교체형 백엔드
- **Sortformer(기본)**: `nvidia/diar_streaming_sortformer_4spk-v2`, **4화자 한정**(`sortformer_backend.py:49`). 청크 단위 추론(`chunk_len=10`×`subsampling_factor=10`×`window_stride`, `:114-118`), 프레임당 argmax 단일화자 → 동시발화 미표현(`:233`). 스트리밍 상태 `spkcache`/`fifo`(각 188)로 청크 간 화자 동일성 유지(`:142-158`).
- **diart(대안)**: pyannote segmentation-3.0 + embedding(`diart_backend.py:166-167`). 누적형, 멀티세션에서 단일 인스턴스 공유 한계(`core.py:268-269`).
- **토큰-화자 정렬(실제 런타임 경로)**: `tokens_alignment.get_lines_diarization()`이 문장(PuncSegment) 단위로 `intersection_duration` 최대 화자 배정(`tokens_alignment.py:182-196`). `diart_backend.add_speaker_to_tokens`는 현재 미사용/레거시로 보임(**확인 필요**).
- **모델 크기**(실측): `.nemo` ≈ 450MB(471,367,680B). torch 디바이스는 cuda/cpu만 — **Apple Silicon MPS/ANE 미사용**(`sortformer_backend.py:62`).
- **요구사항 공백**: '화자 등록/식별/voiceprint' 기능 **전무**(코드 전수 grep 0건). 익명 정수 라벨만 산출. 신규 설계 필요.

### 2.3 Streaming / LocalAgreement
- **LocalAgreement-2 확정 규칙**: 연속 2회 추론에서 동일한 최장 공통 접두사만 commit(`online_asr.py:59-86`, 특히 `:75` 텍스트 동등 비교). 1회만 등장 토큰은 미확정(partial).
- **3단 버퍼**: `committed_in_buffer`/`buffer`/`new`. flush 후 `buffer=new`(`:22-24,83-84`). n-gram(1~5) 중복 제거(`:46-57`).
- **슬라이딩 윈도우 2모드**: `sentence`(문장 토크나이저, **한국어 미지원** `whisper_online.py:39`) / `segment`(ASR segment end 기준, `:300-336`). → **한국어는 `segment` 강제**.
- **프롬프트 캐리오버**: 확정 텍스트 끝 200자를 다음 `init_prompt`로(`:187-209`).
- **조기 확정 옵션**: word probability>0.95면 즉시 commit — **MLXWhisper는 probability 미제공이라 비활성**(`backends.py:209`).
- **동기 블로킹** → `audio_processor.py:408`에서 `asyncio.to_thread`로 오프로드.

### 2.4 ASR 추상화 + SimulStreaming/AlignAtt (MLX)
- **AlignAtt 정책**: cross-attention의 most_attended_frame이 콘텐츠 끝(`content_mel_len - frame <= frame_threshold(4)`)에 도달하면 종료해 안정 prefix만 emit(`align_att_base.py:272-279`).
- **~20개 abstractmethod**(`align_att_base.py:534`)가 **Rust trait 설계의 1:1 청사진**.
- **MLX 두 경로**: 하이브리드(MLX 인코더 + PyTorch 디코더, `backend.py:405-419`)와 `use_full_mlx`(전체 MLX). 단 full MLX는 'punctuation 후 토큰 생성 버그'로 **기본 비활성**(`backend.py:358-363`) → 검증된 경로는 하이브리드뿐.
- **결정적 제약**: AlignAtt는 **디코드 스텝별 cross-attention weight 추출**에 의존(`whisper/model.py:319-323`). 표준 whisper.cpp/mlx-swift는 이를 노출 안 함 → **커스텀 디코더 필요**.
- **UTF-8 부분토큰 처리**: 멀티바이트(한국어) 토큰이 chunk 경계서 잘리면 `�` 감지 후 pending 보류(`align_att_base.py:436-481`) — **한국어 포팅 시 필수 이식**.
- **타임스탬프**: 프레임당 0.02s 하드코딩(TOKENS_PER_SECOND=50, `:242`).

### 2.5 Voxtral MLX 백엔드
- **순수 MLX 재구현**: causal Whisper 인코더 + 2층 어댑터 + Mistral 디코더 + DelayEmbedding(`model.py:440`). PyTorch 의존 없음(토크나이저만 mistral-common).
- **스펙트로그램은 외부 FFT 불필요**: DFT를 cos/sin 행렬곱으로 구현(`spectrogram.py:109`) → Rust 직역 쉬움.
- **시간 동기화 핵심**: `SAMPLES_PER_TOKEN = 1280 = 80ms/토큰`(`spectrogram.py:21`).
- **모델**: `mlx-community/Voxtral-Mini-4B-Realtime-6bit`, 4B 6bit ≈ **3GB(추정, 미검증 — 확인 필요)**.
- **인프로세스 동기 실행**, 환각 방지 2중 안전장치(실제 오디오 안전경계 + 250 position cap, `voxtral_mlx_asr.py:212,40`).
- **토크나이저 종속**: mistral-common Tekkenizer는 Python 전용 → Rust는 tekken.json 파서 직접 구현 필요.

### 2.6 Qwen3 ASR 백엔드
- **qwen3-vllm**: vLLM in-process, **CUDA/Linux 전용**(`pyproject.toml:59-62`) → **온디바이스 부적합**.
- **qwen3-vllm-metal**: vllm-metal MLX, **Darwin arm64 + Python 3.12 전용**(`qwen3_vllm_metal_asr.py:139-144`) → Mac 데스크톱만, iOS 미지원.
- **타임스탬프**: vLLM은 별도 ForcedAligner-0.6B로 정밀 정렬; Metal은 **단어 인덱스 선형 비례 근사**(실제 정렬 아님, `qwen3_vllm_metal_asr.py:349-354`).
- **매 iter 버퍼 전체 재전사** → 버퍼 길이에 비례한 지연/전력 → 실시간성과 충돌.
- **순수 알고리즘 이식 가치**: `_fix_timestamps()` LIS 보간(`:203-263`), `_split_align_words()` CJK/단어 분할(`:166-200`).
- **MLX 단일 워커 스레드 직렬화**(`_Qwen3MetalWorker`, `:55-107`) — Rust에서도 MLX는 전용 스레드/액터로 격리해야 함.
- **모델 크기**(cli 표기): 0.6B≈1.4GB, 1.7B≈3.6GB.

### 2.7 Output protocol + capture + metrics
- **출력 정본**: `FrontData.to_dict()`(`timed_objects.py:196`) — status/lines[]/buffer_*/remaining_time_*. **타임스탬프가 'H:MM:SS.cc' 문자열**로 직렬화됨(`:6,164-165`).
- **speaker 필드 과부하**: -2=침묵, -1→1 매핑, 양수=화자번호(`:162`).
- **full vs diff 두 모드**: 기본 full, diff는 opt-in(`diff_protocol.py:39`). 기본 프론트는 diff 무시(`live_transcription.js:278`).
- **시스템 오디오 캡처는 chrome.tabCapture 전용**(`live_transcription.js:534`) → 네이티브 앱엔 그대로 못 씀.
- **리샘플은 anti-alias 없는 단순 평균**(`recorder_worker.js:27-48`) → 품질 부족.
- **메트릭은 라이브 미전송**: SessionMetrics(rtf/avg/p95 latency)는 cleanup 시 로그로만(`metrics_collector.py:79`). **CPU/메모리는 오프라인 benchmark에만 존재**(`benchmark/runner.py:117`) → 라이브 자원 패널은 **신규 구현**.

### 2.8 Models / config / language
- **정적 매핑 딕셔너리**: `MLX_MODEL_MAPPING`(`model_mapping.py:3`, 13개), cli 카탈로그(`cli.py:142-187`).
- **'로컬에 없으면 자동 다운로드'가 기본**(`model_paths.py:202`). **오프라인 강제 플래그(HF_HUB_OFFLINE 등) 코드에 없음(grep 0건)** → 클라우드 0 정책을 코드 레벨에서 보장 못 함.
- **버전 핀 없음**: snapshot_download가 latest revision을 받음 → 재현성 위험.
- **언어**: 기본 `lan=auto`(자동 감지), ko/en 모두 지원 언어표 포함(`supported_languages.md:7,13`). `.en` 모델이면 lan=en 강제(`config.py:98`).
- **`model_cache_dir`이 MLX/HF 경로에서 무시됨**(전달 안 함, `backends.py:163`) — 버그성 동작(**확인 필요**).
- **warmup이 github에서 JFK wav 다운로드**(`warmup.py:19`) → 오프라인 시 번들 샘플 필요.

---

## 3. 포팅 전략 표

| 서브시스템 | 하는 일 | Rust/Tauri/MLX 매핑 | 전략 | 리스크 |
|---|---|---|---|---|
| **VAD (Silero VAC)** | 음성/침묵 분리, ASR/화자분리 게이트 | `ort` crate로 silero_vad.onnx 실행 + 히스테리시스 상태머신/512프레임 청커는 Rust 직역. 공유 Session + 스트림별 {state,context,triggered,temp_end} | **port_to_rust** (모델은 ort 실행) | 재귀 상태 미보존 시 오동작; 512 프레임 엄격; ONNX 노드명 미검증(확인 필요) |
| **Diarization (분리)** | 화자 구간 산출 + 토큰 정렬 | 오케스트레이션/정렬(`tokens_alignment`, concatenate, intersection)은 Rust 포팅; 모델은 sherpa-onnx/pyannote-onnx를 `ort`로 또는 Sortformer 변환 | **mixed** (정렬=port, 모델=run via onnx/mlx) | NeMo/PyTorch 직접 이식 불가; 4화자 상한; 스트리밍 KV-cache 변환 난이도; 멜 전처리 파라미터 일치 |
| **화자 등록/식별** | (사양 요구, 코드에 없음) | 임베딩 모델(ECAPA/wespeaker, onnx) + 화자 갤러리 DB 신규 설계 | **신규 설계** (참고 대상 없음) | 전체가 신규; 임베딩 영속화/매칭/임계값 설계 필요 |
| **Streaming / LocalAgreement** | 라이브 확정/partial 정책 | HypothesisBuffer + OnlineASRProcessor 전체를 Rust로 1:1(순수 알고리즘). `segment` 트리밍 경로만 | **port_to_rust** | 토큰 텍스트 == 비교가 sep/공백 차이에 취약; 한국어는 segment 강제; segment end 타임스탬프 품질 의존 |
| **ASR 추상화 (AlignAtt)** | 교체형 ASR trait + 저지연 디코드 | abstractmethod → Rust trait. infer 템플릿(슬라이딩윈도우/rewind/UTF-8 pending) Rust 포팅; encode/cross-attn은 MLX | **mixed** | full MLX 디코더 미성숙; 디코드 스텝별 cross-attn 추출 비표준 → 커스텀 디코더 필수 |
| **MLX Whisper 실행** | Whisper 인코더/디코더 추론 | mlx-rs FFI 또는 mlx-swift(iOS). cross-attn 노출 커스텀 빌드 | **run_model_via_mlx** | cross_qk 노출 패치 여부 미확인(확인 필요); 모델 수백MB~1.5GB |
| **Voxtral MLX** | 스트리밍 ASR 모델 + 상태머신 | 상태머신(SlidingKVCache/encode_incremental/안전경계) Rust 포팅; 텐서연산 mlx-rs; 스펙트로그램 Rust 직역; tekken.json 파서 신규 | **mixed** | 4B≈3GB(미검증); mlx-rs API 동등성 미검증; 토크나이저 special token 정합; iOS 부담 |
| **qwen3-vllm** | GPU/vLLM ASR | (버림) | **제외** | Linux/CUDA 전제, 온디바이스 불가 |
| **qwen3-vllm-metal** | MLX ASR (Mac) | Qwen3-ASR을 mlx-rs/mlx-swift로; LIS 보간·CJK 분할·commit 정책 Rust 포팅; MLX 단일 워커 격리 | **run_model_via_mlx** (정책은 port) | vllm-metal 비표준 의존 대체 미정; Metal 타임스탬프 선형 근사(부정확); 전체 재전사 지연; iOS 미지원 |
| **오디오 캡처** | 마이크/시스템오디오 | cpal/AVAudioEngine(마이크), ScreenCaptureKit(Mac 시스템오디오); 리샘플 rubato/AVAudioConverter | **native_apple_api** | iOS 시스템/통화오디오 캡처 정책상 곤란(확인 필요); polyphase 리샘플 권장 |
| **FFmpeg 디코드** | 인코딩 입력 → PCM | pcm_input=True로 우회(네이티브 캡처가 f32 PCM 직접 제공). Mac만 옵션으로 ffmpeg 사이드카 | **native_apple_api** (대부분 제거) | iOS 샌드박스에서 외부 바이너리 spawn 불가 |
| **출력 프로토콜** | FrontData 스트리밍 | WebSocket → Tauri 이벤트(`emit`). FrontData를 serde 구조체로 1:1. diff는 알고리즘만 참고 | **mixed** (스키마 port, diff reference) | 타임스탬프 문자열→float 권장; speaker 매직넘버 → enum |
| **파이프라인 오케스트레이션** | 큐 fan-out/State 병합 | asyncio→tokio. Queue→mpsc, to_thread→spawn_blocking, State→actor. PipelineEvent enum | **port_to_rust** | get_all_from_queue 사적 API 의존 → clean try_recv 재구현; 백프레셔 정책 신설; 타임스탬프 클램핑 정확 이식 |
| **메트릭** | rtf/지연/토큰 수 | SessionMetrics Rust 포팅 + `metrics_update` 이벤트 신설; CPU/RAM은 sysinfo/mach | **mixed** (port + 신규 라이브 채널) | 원본에 라이브 노출 경로 없음; ANE/GPU 사용량 측정 API 불명확 |
| **모델 매니저** | 매핑/다운로드/캐시 | 정적 HashMap + hf-hub/reqwest; detect_model_format std::fs+regex; SHA256 sha2; 캐시 app_data_dir | **port_to_rust** | 버전 핀/오프라인 강제 신규 추가 필요; 캐시 경로 단일화 |
| **config** | 설정 단일 출처 | serde 구조체 + tauri-plugin-store; __post_init__ → validate | **port_to_rust** | .en→lan 등 파생 로직 누락 주의 |

---

## 4. 권장 Rust 라이브 파이프라인 설계

### 4.1 모듈 경계

```
capture (native)         vad (port)            asr (mixed)            diar (mixed)         output (Tauri)
─────────────────        ────────────          ───────────            ──────────           ─────────────
AVAudioEngine/cpal  ─▶   SileroVac        ─▶    StreamingAsrBackend ─▶ State actor    ─▶    emit("transcript_update")
ScreenCaptureKit         (ort+상태머신)         (trait, MLX 뒤)        (tokens+speakers     emit("metrics_update")
 + rubato 16k mono       512프레임 게이트        LocalAgreement        +silence 정렬)        emit("transcript_done")
                                                또는 AlignAtt          DiarBackend(trait)
```

- **capture → vad**: 네이티브 콜백이 f32 PCM(16kHz mono로 리샘플 완료)을 `mpsc::Sender<Vec<f32>>`로 보냄. 이것이 원본 `pcm_input=True` 경로에 해당하며 FFmpeg를 전면 우회.
- **vad → fan-out**: `SileroVac`이 청크를 받아 `Vec<PipelineEvent>`(아래 enum)를 만들고, active 슬라이스를 `transcription_tx`/`diarization_tx` 두 채널로 복제 전송. 침묵은 마커 이벤트로 양 채널에 동일하게 흘려보냄(원본 `_begin_silence`/`_end_silence` 의미 유지).

```rust
enum PipelineEvent {
    Pcm(Vec<f32>),          // 활성 음성 슬라이스 (절대시각 메타 포함)
    SilenceStart { at: f64 },
    SilenceEnd   { duration: f64 },
    ChangeSpeaker { at: f64 },
    Sentinel,               // end-of-stream (원본 SENTINEL)
}
```

> 원본의 `isinstance(Silence/ChangeSpeaker/np.ndarray)` 분기(`audio_processor.py:382-401`)를 enum match로 치환하는 것이 가장 큰 명료성 이득. 모든 dispatch 지점을 enum variant로 열거해 누락 방지.

### 4.2 스레딩 (tokio)

| 원본 | Rust |
|---|---|
| `asyncio.Task` (consumer) | `tokio::spawn` |
| 블로킹 추론 `asyncio.to_thread` | `tokio::task::spawn_blocking` (MLX는 **전용 스레드/액터**로 격리 — Qwen Metal 워커 패턴) |
| `asyncio.Queue` | `tokio::sync::mpsc` (모달리티당 1채널) |
| `State` + `asyncio.Lock` | **State actor**: State를 한 태스크가 소유, 갱신을 채널로 수신 (락 경합 회피) |
| `results_formatter` (50ms 폴링) | tokio 태스크 → broadcast/mpsc → `app.emit("transcript_update")` |
| 전역 `threading.Lock` (`thread_safety.py:34`) | **제거** (단일 사용자) |

MLX 호출은 반드시 **단일 워커 스레드**에 직렬화(`mx.set_default_device` 안전성, `qwen3_vllm_metal_asr.py:55-107` 근거). Rust에서는 dedicated thread + 요청 채널(액터)로 구현하고, 다른 tokio 태스크는 oneshot으로 결과를 await.

### 4.3 백프레셔

- 원본 큐는 **unbounded asyncio.Queue** → 느린 온디바이스 추론에서 메모리 무한 증가 위험(원본도 바이트 버퍼만 경고, `audio_processor.py:763`).
- Rust는 **bounded mpsc**(예: capacity = 수 초 분량)로 바꾸되, `send().await`가 막히면 의미가 달라지므로 **명시적 drop/병합 정책** 필요:
  - VAD→ASR: 오디오는 손실 금지. 채널이 차면 capture 콜백이 백프레셔를 받아 OS 버퍼에 쌓이게 두거나, 가장 오래된 활성 슬라이스를 병합(`get_all_from_queue`의 concat batching 의미를 try_recv 루프로 재구현, **사적 `_queue` 접근은 금지**).
  - ASR→output: 최신 우선. broadcast로 마지막 스냅샷만 유지(coalesce).
- `get_all_from_queue` 배칭(`audio_processor.py:28`)은 **Silence/Sentinel 경계에서 멈추는** clean try_recv 루프로 재현. 경계 의미를 깨면 타임스탬프 드리프트 발생.

### 4.4 타임스탬프 정합 (고위험)
- 원본은 producer의 sample-precise total과 consumer의 cumulative stream time을 분리 추적하며 클램핑으로 드리프트를 막는다(`audio_processor.py:798-810`). Rust 포팅에서 이 클램핑과 침묵 누적 순서를 정확히 이식하지 않으면 긴 회의에서 누적 드리프트. **TokensAlignment(`tokens_alignment.py`)는 별도 분석 후 이식**(본 분석 범위 밖, 확인 필요).

---

## 5. 교체형 ASR 백엔드 트레이트 설계

원본 `ASRBase`(`backends.py:15`)와 AlignAtt abstractmethod(`align_att_base.py:534`)가 근거. 두 계층으로 분리한다.

### 5.1 상위 trait — 스트리밍 정책이 호출하는 계약

```rust
pub struct AsrToken { pub start: f64, pub end: f64, pub text: String,
                      pub probability: Option<f32>, pub detected_language: Option<String> }

/// LocalAgreement / AlignAtt 등 정책 계층이 호출.
pub trait StreamingAsrBackend: Send {
    fn insert_audio_chunk(&mut self, pcm: &[f32], end_time: f64);
    /// 1회 추론 → 확정 토큰. is_last면 잔여 flush.
    fn process_iter(&mut self, is_last: bool) -> Result<Vec<AsrToken>, AsrError>;
    fn get_buffer(&self) -> String;          // 미확정 partial
    fn start_silence(&mut self, at: f64);    // 침묵/리셋 훅
    fn on_change_speaker(&mut self, at: f64);// 화자 전환 시 컨텍스트 리셋
    fn finish(&mut self) -> Vec<AsrToken>;
    fn warmup(&mut self);
    fn set_language(&mut self, lang: Option<&str>); // auto=None (코드스위칭)
}
```

> LocalAgreement를 쓰는 백엔드(MLX Whisper)와 자체 스트리밍 백엔드(Voxtral/Qwen)를 **동일 trait 뒤**에 둔다. LocalAgreement용 백엔드는 더 얇은 trait(`transcribe`/`ts_words`/`segments_end_ts`)을 추가로 구현하고, `OnlineASRProcessor`(LocalAgreement 정책)가 그 위에 올라간다.

```rust
/// LocalAgreement 정책이 감싸는 "1회 추론" 백엔드 (backends.py:15 등가)
pub trait WhisperLikeBackend: Send {
    fn transcribe(&self, audio: &[f32], init_prompt: &str) -> AsrResult;
    fn ts_words(&self, r: &AsrResult) -> Vec<AsrToken>;     // 단어 타임스탬프 필수
    fn segments_end_ts(&self, r: &AsrResult) -> Vec<f64>;   // segment 트리밍용
    fn sep(&self) -> &str;                                  // ' ' vs '' (토큰 결합)
}
```

### 5.2 프레임워크 텐서 훅 (AlignAtt용, 선택)

```rust
/// AlignAtt 디코드 루프(순수 제어흐름)가 호출하는 텐서 추상화.
pub trait AlignAttHooks {
    fn encode(&mut self, pcm: &[f32]) -> (Features, usize /* content_mel_len */);
    fn logits_and_cross_attn(&mut self, tokens: &[u32], feats: &Features) -> (Logits, CrossAttn);
    fn get_attended_frames(&self, attn: &CrossAttn, content_len: usize) -> usize;
}
```

### 5.3 각 백엔드의 온디바이스 실행 경로

| 백엔드 | Mac 경로 | iOS 경로 | 핵심 검증(확인 필요) |
|---|---|---|---|
| **MLX Whisper** | mlx-rs FFI 또는 Python+mlx 사이드카; LocalAgreement는 단어 타임스탬프 필요 | mlx-swift(Rust↔Swift FFI) | cross-attn 노출 커스텀 빌드 여부(AlignAtt 쓸 때); ts_words 제공 여부 |
| **Voxtral (MLX)** | 1단계 사이드카 PoC → 2단계 상태머신 Rust + mlx-rs 텐서 | mlx-swift 전면 재구현(4B≈3GB 부담) | mlx-rs의 fast.rope/sdpa/conv1d/quantize(6bit) 동등성; tekken.json 토큰 ID |
| **qwen ASR (Metal)** | mlx-rs/mlx-swift로 Qwen3-ASR; 정책은 Rust 포팅 | mlx-swift(0.6B 권장) | vllm-metal 내부 그래프 미확인; CoreML 변환 가능성 |

**권장 진행**: (1) Mac에서 **사이드카(Python+MLX) PoC**로 동작 확보 → (2) 스트리밍 상태머신/정책을 Rust로 이전, 텐서연산만 MLX 호출로 좁힘 → (3) iOS는 mlx-swift 네이티브. 사이드카는 **iOS에서 불가**(별도 프로세스/Python 런타임 불가).

---

## 6. 온디바이스 / 오프라인 · 자원 모니터링

### 6.1 클라우드 0 보장 지점
- **모든 추론은 로컬 모델 가중치 기반** — 원본도 런타임 클라우드 추론 0(HF는 다운로드만). 단 **OpenAI API 백엔드(`backends.py:235`)와 qwen3-vllm은 제외 대상**.
- **위험**: 원본에 오프라인 강제 플래그가 없어(`HF_HUB_OFFLINE` grep 0건), '로컬에 없으면 자동 다운로드'가 기본(`model_paths.py:202`). Rust 포팅 시 **다운로드 단계와 추론 단계를 명시적으로 분리**하고, 추론 경로에서는 네트워크 접근을 코드 레벨로 차단(예: 로컬 경로만 허용, 미존재 시 명시적 에러).

### 6.2 모델 다운로드 / 캐시
- Rust 모델 매니저: 정적 카탈로그(`model_catalog.json`) + `hf-hub` crate 또는 `reqwest`로 HF resolve API 직접 호출.
- **버전 핀 추가 권장**: rev를 commit SHA로 고정(원본은 latest, 재현성 위험).
- **무결성 검증**: OpenAI 경로의 SHA256(`whisper/__init__.py:57`)을 `sha2` crate로 전 경로에 확대.
- **캐시 단일화**: `~/.cache/huggingface` 의존을 끊고 Tauri `app_data_dir` 하위로 통일(샌드박스/오프라인 정책 부합). 원본의 `model_cache_dir` 무시 버그도 함께 정리.
- **warmup**: github JFK wav 대신 **번들 샘플 PCM**으로 1회 추론(`warmup.py` 대체).

### 6.3 자원(메모리/CPU/지연) 실시간 표시
- **원본에 라이브 소스 없음** — 신규 구현 필수.
- 측정: SessionMetrics(rtf, avg/p95 latency, queue depth, n_tokens) Rust 포팅(`metrics_collector.py` 기반) + **`metrics_update` Tauri 이벤트를 1초 주기로 emit**.
- CPU/RAM: `sysinfo` crate 또는 mach `task_info`(RSS).
- **GPU/ANE(Neural Engine) 사용량**: 측정 API/권한이 불명확(IOReport/powermetrics는 권한 이슈) → **근사치만 가능(확인 필요)**. 초기엔 CPU/RAM/지연/RTF만 노출하고 ANE는 후순위.

---

## 7. 한·영 이중언어 (코드스위칭)

- **지원**: ko/en 모두 지원 언어표 포함(`supported_languages.md:7,13`). 기본 `lan=auto`(자동 감지, `parse_args.py:120`).
- **LocalAgreement/정렬 로직은 언어 무관**(텍스트 토큰 비교) — 코드스위칭 품질은 **백엔드 모델 능력 문제로 분리**됨.
- **한국어 필수 이식 포인트**:
  - **`segment` 트리밍 강제**: 문장 토크나이저가 한국어 미지원(`whisper_online.py:39`).
  - **UTF-8 부분토큰 pending 처리**(`align_att_base.py:436-481`): 멀티바이트 한글이 chunk 경계서 잘릴 때 `�` 보류. Rust 포팅 시 반드시 이식.
- **언어 감지 단위**: 원본은 **발화/라인 단위** 언어 감지만(`detected_language`는 line 단위, `timed_objects.py:169`). **한 문장 내 혼용 표시 정책은 미정(확인 필요)**.
- **코드스위칭 정확도**: MLX Whisper(multilingual), Voxtral, Qwen3-ASR 각 모델의 실제 한·영 혼용 정확도는 **미검증(확인 필요)** — 가중치를 열지 않음. SimulStreaming 기본 config가 `language='zh'`(`config.py:14` 추정)인 점도 한·영 재튜닝 필요 신호.

---

## 8. iOS 특이사항

### 8.1 시스템/통화 오디오 캡처 (최대 난제)
- **macOS**: ScreenCaptureKit(`SCStream` audio)로 시스템 오디오 캡처 가능(화면녹화 권한 필요). 마이크는 `AVAudioEngine` + `NSMicrophoneUsageDescription`.
- **iOS**: 임의 시스템/통화 오디오 캡처는 OS 정책상 **사실상 불가(확인 필요)**.
  - ReplayKit/Broadcast Upload Extension은 **자체 앱/브로드캐스트 오디오 한정**, 통화 오디오 접근 불가로 추정.
  - 가능한 우회: (a) 마이크만으로 시작(스피커 출력을 마이크가 주워담는 환경), (b) Broadcast Upload Extension으로 화면 공유 중 앱 오디오 캡처(통화는 제외), (c) CallKit/통화 녹음은 별도 정책 검토.
  - **결론**: iOS는 1차로 **마이크 입력만** 지원하고 시스템/통화 오디오는 별도 PoC/정책 조사 대상.

### 8.2 iOS 불가 / 부담 요소
- **FFmpeg subprocess**: iOS 샌드박스에서 외부 바이너리 spawn 불가 → **pcm_input 경로로 전면 우회**(네이티브 캡처 + Rust 리샘플).
- **vLLM(qwen3-vllm)**: Linux/CUDA 전제 → iOS 불가, 제외.
- **vllm-metal(qwen3-vllm-metal)**: Darwin arm64 데스크톱 휠 + Python 3.12 락인 → **iOS 미지원**. iOS는 mlx-swift 재구현 필요(확인 필요).
- **PyTorch/NeMo(Sortformer), CTranslate2(faster-whisper)**: iOS 비현실적 → MLX/onnx 경로로 단일화.
- **사이드카(Python+MLX) 전략**: iOS 원천 불가. Mac PoC 전용.
- **모델 크기**: Voxtral 4B(~3GB 추정), Whisper large(~1.5GB), Qwen 1.7B(3.6GB)는 iOS 메모리/스토리지 압박 → iOS는 **소형 모델(Whisper tiny/small, Qwen 0.6B) 우선**, On-Demand Resources/App Group 다운로드 고려.

---

## 9. 우선순위 오픈 퀘스천 (중요도 순)

1. **iOS 시스템/통화 오디오 캡처 가능성** — ReplayKit/Broadcast Extension/CallKit으로 어디까지 되는지. 안 되면 iOS 제품 범위(마이크 전용)를 재정의해야 함. **제품 범위를 좌우하는 최상위 결정.**
2. **MLX Whisper(및 Voxtral/Qwen)가 디코드 스텝별 cross-attention(cross_qk)을 노출하는가** — AlignAtt 성립 여부의 핵심. mlx-swift 커스텀 빌드 vs whisper.cpp 커스텀 vs CoreML 변환 PoC 필요. 노출 불가면 AlignAtt 대신 LocalAgreement 단일화.
3. **온디바이스 ASR 백엔드 실행 경로 확정** — mlx-rs(성숙도) vs Swift FFI(mlx-swift) vs 사이드카. Mac/iOS 각각의 1차 경로 결정. 이후 모든 백엔드 trait 구현의 전제.
4. **mlx-rs API 동등성** — fast.rope(traditional)/scaled_dot_product_attention/conv1d/nn.quantize(6bit)를 Rust에서 동일 수치로 재현 가능한가(Voxtral/Qwen 포팅 가능성의 관건).
5. **화자 등록/식별 신규 설계** — 임베딩 모델 후보(ECAPA-TDNN/wespeaker, onnx 가용)와 화자 갤러리/매칭/임계값. 원본에 참고 코드 0건이라 처음부터 설계.
6. **온디바이스 화자분리 모델 대체** — Sortformer(PyTorch/450MB, 4화자 상한, 동시발화 미표현)를 sherpa-onnx vs pyannote-onnx vs Sortformer 변환 중 무엇으로. 한·영 회의·4화자 초과·동시발화 벤치마크 필요.
7. **클라우드 0 / 오프라인 강제를 Rust에서 어떻게 코드 보장** — 다운로드/추론 단계 분리, 버전 SHA 핀, 캐시 단일화, 추론 경로 네트워크 차단.
8. **타임스탬프 정합(TokensAlignment) 이식** — `tokens_alignment.py`(본 분석 범위 밖)의 토큰⨉화자⨉침묵 정렬과 클램핑 로직을 정확히 이식해야 드리프트 방지. 별도 정밀 분석 필요.
9. **자원 모니터 데이터 소스** — CPU/RAM은 sysinfo로 가능하나 ANE/GPU 사용량 측정 API·권한 불명확. 1차 노출 항목(RTF/지연/CPU/RAM) 확정.
10. **모델별 한·영 코드스위칭 실측 정확도** — MLX Whisper/Voxtral/Qwen3-ASR 각각. 모델 가중치/벤치마크 외부 확인 필요. 기본 언어 설정(SimulStreaming `zh` 추정) 재튜닝 여부.
11. **Silero ONNX 그래프 노드명 검증** — 'input'/'state'/'sr' 및 출력명을 `ort` 연결 시 실제 그래프로 확인(호출 코드 기반 추론값).
12. **Voxtral/Qwen 모델 실제 디스크/RAM 크기** — 4B 6bit≈3GB는 추정. iOS 적재 가능 모델 상한 결정에 필요.
