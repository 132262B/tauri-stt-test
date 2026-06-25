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

## 알려진 한계 / 다음 작업

- **화자 분리가 Rust 경로엔 미적용**: resemblyzer(Python)라 Voxtral/Qwen(사이드카)에서만 화자 라벨. **Rust 화자분리(ort + onnx 임베딩 + 클러스터링) 포팅**이 다음 핵심(Whisper에도 화자 라벨 부여).
- **Voxtral/Qwen 순수 Rust 경로 없음**: 현재 Python 사이드카. all-Rust 원하면 후속 연구 필요(mlx-rs/candle).
- **사이드카 배포 번들**: dev venv python 직접 spawn. 배포는 PyInstaller + 코드서명(미구현).
- **시스템 오디오(Mac ScreenCaptureKit), iOS** 미착수.
- **품질 튜닝**: 자동 언어감지 ko↔en 플립(언어 고정으로 완화), 무음 끝 반복 환각.

## 권장 다음 순서

1. **Rust 화자분리**(ort + onnx 임베딩) — Whisper 경로에도 화자 라벨.
2. 시스템 오디오(ScreenCaptureKit) + 믹서.
3. 사이드카 PyInstaller 번들 + 코드서명(배포).
