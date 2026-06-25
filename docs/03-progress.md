# 03 — 진행 상황 (자율 작업 세션)

> 사양/아키텍처는 `00`/`01`/`02` 참고. 이 문서가 현재 실제 상태.

## 한 줄 요약

Mac에서 **마이크 → 실시간 화자/평문 전사 → 화면 + 자원 모니터 + 내보내기**가 동작한다.
**Whisper 경로는 완전 Rust 네이티브**(whisper.cpp/Metal, in-process, **Python 0**). Voxtral·Qwen3-ASR은
순수 Rust 온디바이스 구현이 없어 **Python 사이드카**로 제공(선택). 영어·한국어·2화자·오프라인 검증 완료.

## ASR 백엔드 (모델 드롭다운)

| 모델 | 엔진 | 실행 | 화자분리 |
|---|---|---|---|
| Whisper turbo/large-v3/small/base/tiny (`ggml-*`) | **whisper.cpp (Rust, Metal)** | **in-process Rust, Python 0** | 미적용(아래 참고) |
| Qwen3-ASR 1.7B | qwen3-asr-mlx | Python 사이드카 | 적용(resemblyzer) |
| Voxtral Mini 3B | mlx-voxtral | Python 사이드카 | 적용(resemblyzer) |

- Whisper는 `WhisperStreamingBackend`(`crates/stt-asr-whisper`) + Rust `OnlineAsrProcessor`(LocalAgreement-2, `stt-core`).
  ggml 모델은 HF(ggerganov/whisper.cpp)에서 자동 다운로드(`.hf-cache/ggml`).
- Voxtral/Qwen은 `crates/stt-asr-sidecar`가 Python(`sidecar/stt_mlx`)을 spawn(stdin/stdout NDJSON + UDS PCM).
- 모든 설치는 프로젝트 내부(venv·모델캐시·cmake). 전역 0.

## 검증 (헤드리스)

- **Rust 네이티브 Whisper**: `cargo test -p stt-asr-whisper --test whisper_streaming -- --ignored` → JFK 전사 정확(Python 0, Metal).
- **LocalAgreement Rust 포팅**: `cargo test -p stt-core` 단위테스트 통과.
- **내보내기 포맷터**: `cargo test --lib export` 5개 통과.
- **사이드카(Qwen/Voxtral)**: 자가 self-test 로 JFK 정확 전사 확인.
- **한국어·2화자·오프라인(HF_HUB_OFFLINE)**: 사이드카 경로에서 확인(화자 0/1 재식별 포함).

## 직접 실행

```bash
cd app && pnpm install
export PATH="$HOME/.cargo/bin:$PATH"   # cargo PATH
pnpm tauri dev
# 모델 선택(기본=Whisper turbo, Rust) → "전사 시작" → 마이크 권한 → 전사 누적
```
첫 시작은 모델 최초 다운로드. cmake 필요 시 `app/src-tauri/sidecar/.venv/bin` 을 PATH에.

## Seed 항목별 상태 (전부 Rust/Tauri, Python·Node 런타임 0)

| Seed 항목 | 상태 |
|---|---|
| 온디바이스/클라우드0 (C1) | ✅ whisper.cpp 로컬, 추론 네트워크 0 |
| Rust/Tauri, Mac (C2) | ✅ (iOS 미착수) |
| 실시간 라이브 전사 (C5) | ✅ LocalAgreement Rust |
| **화자 라벨 전사 (C6)** | ✅ **sherpa-onnx CAM++ 온라인 화자분리(Rust)** — 2화자 구분·재식별 검증 |
| 화자 등록·식별 (C13) | ⚠️ 클러스터링/재식별은 됨. 사용자 이름 등록 UI는 미구현 |
| VAD (C4) | ⚠️ 게이트 trait·로직은 마련. sherpa-rs Silero가 SIGSEGV라 보류 → whisper.cpp 무음 처리로 대체 |
| 한·영 (C10) | ✅ multilingual + 언어 고정 |
| 자원 모니터 | ✅ | 내보내기(C11) ✅ | 저지연 ✅ |
| 교체형 백엔드 (C3) | ⚠️ Whisper 5종(Voxtral/Qwen은 Python 필요 → no-Python 규칙으로 제외) |
| **시스템 오디오 (C7)** | ✅ 구현(macOS ScreenCaptureKit, Rust). 입력 선택 mic/system/both + 믹서. **컴파일·부팅 검증**, 런타임은 화면녹화 권한 필요(미검증) |
| iOS (C2) | ❌ 미착수 (디바이스 필요) |
| 실패·엣지 (C12) | ⚠️ 부분 |

## 다음 (헤드리스 검증 가능/불가 구분)

- 검증 가능: 명시적 Silero VAD 게이트, 화자 이름 등록.
- 디바이스/권한 필요(헤드리스 검증 불가): 시스템 오디오(ScreenCaptureKit 권한), iOS(Xcode/실기기).
