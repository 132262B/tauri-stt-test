# 03 — 진행 상황 (자율 작업 세션)

> 이 세션에서 자율로 진행한 결과 요약. 사양/아키텍처는 `00`/`01`/`02` 참고.

## 한 줄 요약

**Mac에서 마이크 → 온디바이스 MLX Whisper 실시간 전사(확정/partial) → 화면 표시 + 자원 모니터 + txt/srt/json 내보내기**가 동작한다. 영어·한국어 전사와 오프라인(클라우드 0) 동작을 헤드리스로 검증했다. P0 스캐폴드 + P1 핵심(커밋4·5~9·12·13) 완료.

## 완료 (커밋 순)

| 커밋 | 내용 | 검증 |
|---|---|---|
| docs 00~02 | 사양·whisper-live-kit 분석·확정 아키텍처 | — |
| build | src-tauri 워크스페이스화 + stt-core/proto 크레이트 | cargo check |
| feat | shell·store·dialog 플러그인 + capabilities(데스크톱/모바일) | cargo+pnpm build |
| feat | greet 제거 + commands/events/app_state 스텁 + 프론트 셸 | build |
| feat(capture) | cpal 마이크 → rubato 16k mono → mpsc | build |
| feat(sidecar) | **Python+MLX 사이드카**: LocalAgreement-2·MLX Whisper 어댑터(wlk 이식) | **JFK 자가전사 통과** |
| feat(asr) | stt-core ASR trait·드라이버 + **Rust 사이드카 백엔드**(spawn+UDS+NDJSON) | **통합테스트 통과(34s)** |
| feat(session) | 마이크→사이드카→`transcript_update` 배선 + 프론트 렌더 | 풀 빌드 링크 |
| feat(metrics) | 자원 모니터(CPU/RSS/RTF/지연) → `metrics_update` + 패널 | build |
| feat(export) | txt/srt/json 내보내기 + 확정 토큰 누적 | **단위테스트 4개 통과** |
| feat(macos) | 마이크 권한 Info.plist | — |

## 검증 결과 (헤드리스)

- **영어**: JFK 샘플 → `"And so my fellow Americans ask not what your country can do for you. ask what you can do for your country."` (정확)
- **한국어**: `say -v Yuna` 생성 샘플 → `"안녕하세요 오늘 회의를 시작하겠습니다 이번 프로젝트의 핵심은 온디바이스 … 인식입니다"` (정확. 영어 "STT"는 TTS 발음을 모델이 "stp"로 인식)
- **오프라인(AC1)**: `HF_HUB_OFFLINE=1` 캐시 전용 → 정확 전사. **추론 시 네트워크 0 확인.**
- **스트리밍**: committed(확정)/buffer(partial) 점진 갱신 동작.
- **Rust↔사이드카 통합테스트**: `cargo test -p stt-asr-sidecar --test jfk_transcribe -- --ignored` 통과.

## 직접 실행해 확인할 것 (GUI는 헤드리스 불가)

```bash
cd app && pnpm install && pnpm tauri dev
# "● 전사 시작 (마이크)" 클릭 → 마이크 권한 허용 → 말하면 전사가 화면에 누적
# 우측 자원 모니터(메모리/CPU/RTF) 1초 갱신, 하단 내보내기(txt/srt/json)
```
첫 시작은 모델(turbo ~1.5GB, 최초 1회) 다운로드/로딩으로 수 초~수 분.

## 알려진 한계 / 다음 작업

- **화자 분리 미구현** (P1.5): 현재 화자 라벨 없음. onnx diar + 트랙 클러스터링 + 등록/식별 신규 설계 필요(아키텍처 I).
- **시스템 오디오 캡처 미구현** (P1.5): 현재 마이크만. Mac은 ScreenCaptureKit 예정.
- **사이드카 번들링 미완** (배포): 현재 dev venv python 직접 spawn. 배포는 PyInstaller 단일 바이너리 + 코드서명/공증 필요(아키텍처 최상위 리스크, 커밋5 노트).
- **백엔드 전환 미구현** (P2): 현재 MLX Whisper 고정. Voxtral/qwen 백엔드 + hot-swap은 trait 뒤에 drop-in 예정(아키텍처 D).
- **네이티브 MLX(mlx-rs) 미전환** (P2): 정책/VAD는 아직 사이드카 내부(Python). Rust 이전 + 골든 회귀 예정.
- **iOS 미착수** (P3): 마이크 전용 예정.
- **전사 품질 튜닝**: 긴 무음/문장 끝에서 Whisper 반복 환각(`ö ö ö`) 관측 — `condition_on_previous_text`/no_speech 필터 튜닝 대상.
- **모델 매니저/오프라인 강제 코드화** (커밋10/11): 현재 HF 캐시 의존. 카탈로그·SHA 핀·다운로드 UI·오프라인 게이트는 미구현(오프라인 동작 자체는 검증됨).

## 권장 다음 순서

1. 화자 분리(P1.5) — 사용자가 "화자 라벨 전사"를 핵심 산출물로 정의함(AC: 2명+ 구분).
2. 시스템 오디오(Mac ScreenCaptureKit) + 믹서.
3. 사이드카 PyInstaller 번들 + 코드서명(배포 가능성 검증, 최상위 리스크).
4. 백엔드 전환(Voxtral/qwen) + 네이티브 MLX 전환.
