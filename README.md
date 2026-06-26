# tauri-stt-test

**온디바이스(클라우드 0) 실시간 회의 전사** Tauri 2 앱. 백엔드는 Rust, ASR 추론은
Apple Silicon **MLX Whisper**를 쓰며 모든 추론이 로컬에서 일어난다(모델만 인터넷 다운로드).
VAD·화자분리·LocalAgreement 스트리밍 정책은 [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)을 **참고/이식**해 개발한다.

> 사양·분석·아키텍처·진행상황은 `docs/00~03` 참고. 우선순위는 **Mac → iOS**.

## 구조

```
tauri-stt-test/
├─ app/                       # Tauri 2 앱
│  ├─ src/                    # 프론트(React+TS): 전사 뷰·자원 모니터·내보내기
│  └─ src-tauri/
│     ├─ src/                 # 얇은 Tauri 어댑터(commands/events/session/capture)
│     ├─ crates/
│     │  ├─ stt-core/         # tauri 무의존 순수 코어: ASR trait·LocalAgreement 드라이버·메트릭·출력
│     │  ├─ stt-sidecar-proto/# Rust↔Python 사이드카 메시지/프레임 스키마
│     │  └─ stt-asr-sidecar/  # StreamingAsrBackend 사이드카 구현(프로세스+UDS+NDJSON) + 통합테스트
│     ├─ sidecar/             # Python+MLX 사이드카(stt_mlx) — venv·모델캐시는 프로젝트 로컬
│     ├─ Info.plist           # macOS 마이크 권한
│     └─ gen/{apple,android}  # 모바일 프로젝트(자동생성)
├─ whisper-live-kit/          # 참고용 원본(Python). 직접 실행하지 않음 — 코드 이식 출처.
└─ docs/                      # 00 사양 · 01 분석 · 02 아키텍처 · 03 진행상황
```

## 동작 (현재)

마이크 → cpal 캡처 → 16kHz mono 리샘플 → ASR → 실시간 전사(확정/partial) → 화면 + 자원 모니터 + txt/srt/json 내보내기.

ASR 백엔드(모델 드롭다운에서 선택):
- **Whisper (`ggml-*`)** — **완전 Rust 네이티브**: `whisper.cpp`(Metal) + Rust LocalAgreement, **Python 0, in-process**. 기본값.
- **Qwen3-ASR / Voxtral** — Python 사이드카(순수 Rust 온디바이스 구현 부재). 화자 분리(resemblyzer)는 현재 이 경로에서만.

한국어·영어·2화자·오프라인(HF_HUB_OFFLINE) 검증됨. 모든 설치는 프로젝트 내부(venv·모델캐시·cmake), 전역 0.

## 사전 준비물 (이 맥에는 모두 설치 완료)

- Node 26 / pnpm 10, Rust(rustup) + iOS/Android 타겟, Xcode + CocoaPods
- `cargo`가 비대화형 셸 PATH에 없으면: `export PATH="$HOME/.cargo/bin:$PATH"`

## 사이드카(Python+MLX) 준비 — 프로젝트 로컬

```bash
cd app/src-tauri/sidecar
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export HF_HOME="$PWD/.hf-cache"   # 모델 캐시도 프로젝트 내부

# 헤드리스 전사 검증(모델 최초 1회 다운로드)
./.venv/bin/python -m stt_mlx.main --self-test test-data/jfk.wav \
    --model mlx-community/whisper-large-v3-turbo
```

## 실행

루트에서 바로 실행:

```bash
./run-app              # 기본: Q5 CoreML OFF, GPU env OFF (메모리 절약)
make dev               # 위와 동일
./run-app --coreml     # Q5 CoreML encoder 실험용(메모리 증가)
./run-app --gpu        # whisper.cpp GPU/Metal env 실험용
```

기존 방식도 그대로 가능:

```bash
cd app
pnpm install
pnpm tauri dev          # macOS 데스크톱 — "전사 시작" 클릭 후 마이크 권한 허용
pnpm tauri build        # 릴리스 .app / .dmg
pnpm tauri ios dev      # iOS (P3, 마이크 전용 예정)
```

> 현재 사이드카는 개발 중 `app/src-tauri/sidecar/.venv`의 python 을 직접 spawn 한다
> (PyInstaller 단일 바이너리 번들 + 코드서명은 후속). 첫 `전사 시작`은 모델 로딩에 수 초 걸린다.

## 테스트

```bash
cd app/src-tauri
export PATH="$HOME/.cargo/bin:$PATH"
cargo test --lib export                                   # 내보내기 포맷터 단위테스트
cargo test -p stt-asr-sidecar --test jfk_transcribe -- --ignored --nocapture  # 사이드카 전체 전사(venv+모델 필요)
```
