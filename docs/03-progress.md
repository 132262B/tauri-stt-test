# 03 — 진행 상황 (자율 작업 세션)

> 이 세션에서 자율로 진행한 결과 요약. 사양/아키텍처는 `00`/`01`/`02` 참고.

## 한 줄 요약

**Mac에서 마이크 → 온디바이스 MLX Whisper 실시간 전사(확정/partial) + 온라인 화자 분리(화자별 라인) → 화면 표시 + 자원 모니터(메모리/CPU/RTF) + txt/srt/json 내보내기 + 모델 선택**이 동작한다. 영어·한국어·2화자·오프라인(클라우드 0)을 헤드리스로 검증했다. P0 + P1 핵심 + P1.5 화자분리 완료.

## 동작하는 기능

- 🎙 **마이크 캡처**(cpal) → 16kHz mono 리샘플(rubato)
- 🧠 **온디바이스 ASR**: Python+MLX 사이드카(MLX Whisper), LocalAgreement-2 스트리밍(확정/partial)
- 👥 **온라인 화자 분리**: resemblyzer d-vector + 코사인 클러스터링, 청크 간 화자 재식별, 화자별 라인 렌더
- 📊 **자원 모니터**: 앱+사이드카 RSS·CPU·RTF·지연(p50/p95) 1초 갱신
- 💾 **내보내기**: txt/srt/json(화자 라벨 포함)
- 🔀 **모델 선택**: large-v3-turbo/large-v3/small/base/tiny 교체(속도·메모리 비교)
- 🔌 **클라우드 0**: 추론 시 네트워크 0(검증), 모델만 인터넷 다운로드

## 검증 결과 (헤드리스)

- **영어**(JFK): `"...ask not what your country can do for you..."` 정확.
- **한국어**(`say` 합성): `"안녕하세요 오늘 회의를 시작하겠습니다 …"` 정확.
- **2화자**(Yuna→Daniel→Yuna 합성): spk0/spk1 구분 + Yuna 재등장 시 **spk0 재식별**.
- **오프라인(AC1)**: `HF_HUB_OFFLINE=1` 캐시 전용 → 정확 전사. 추론 네트워크 0.
- **테스트**: export 포맷터 단위테스트 5개 통과, Rust↔사이드카 통합테스트(JFK) 통과(~35s).

## 직접 실행해 확인 (GUI는 헤드리스 불가)

```bash
cd app && pnpm install && pnpm tauri dev
# 모델 선택 → "● 전사 시작" → 마이크 권한 허용 → 화자별 전사 누적
# 우측 자원 모니터(메모리/CPU/RTF), 하단 내보내기(txt/srt/json)
```
첫 시작은 모델 최초 다운로드/로딩으로 수 초~수 분(이후 캐시).

## 커밋 이력(이 세션)

docs(00~03) · build(워크스페이스) · feat(플러그인/스텁/캡처/사이드카/ASR배선/메트릭/내보내기/화자분리/모델선택) · feat(macOS 마이크 권한). 모든 커밋에 AI 도구 표기 없음.

## 알려진 한계 / 다음 작업

- **사이드카 번들링(배포)**: 현재 dev venv python 직접 spawn. 배포는 PyInstaller 단일 바이너리 + 코드서명/공증 필요(아키텍처 최상위 리스크). → 데스크톱 배포 전 필수.
- **시스템 오디오 캡처(Mac)**: 현재 마이크만. ScreenCaptureKit + 믹서(B.6) 미구현.
- **백엔드 전환(Voxtral/qwen)**: 현재 MLX Whisper 고정. trait 뒤 drop-in 예정(아키텍처 D).
- **네이티브 MLX(mlx-rs) 전환**: 정책/VAD는 아직 사이드카(Python). Rust 이전 + 골든 회귀(J.1) 예정.
- **iOS(P3)**: 미착수(마이크 전용 예정).
- **화자 reconciliation/소급 정정**: 현재 트랙 즉시 배정만. 라이브 소급 relabel(I.1)·등록/식별(이름 매칭)은 미구현.
- **모델 매니저/오프라인 강제 코드화(커밋10/11)**: 현재 HF 캐시 의존. 카탈로그·SHA 핀·다운로드 UI·오프라인 게이트 미구현(오프라인 동작 자체는 검증됨).
- **전사 품질**: 자동 언어감지가 화자/문맥 전환 시 ko↔en 플립, 무음 끝 반복 환각(`ö ö ö`) 관측 — `condition_on_previous_text`/언어 고정/no_speech 필터 튜닝 대상.

## 권장 다음 순서

1. 화자 등록/식별(이름 매칭) + 소급 relabel — "누가" 회의록의 완성도.
2. 시스템 오디오(Mac ScreenCaptureKit) + mic 믹싱.
3. 사이드카 PyInstaller 번들 + 코드서명(배포 가능성, 최상위 리스크).
4. 백엔드 전환(Voxtral/qwen) + 네이티브 MLX 전환(골든 회귀).
