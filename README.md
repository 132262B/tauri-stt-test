# tauri-stt-test

Tauri 2 기반 크로스플랫폼 STT(실시간 음성→텍스트) 앱. 백엔드는 [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit).

## 구조

```
tauri-stt-test/
├─ app/                 # Tauri 2 앱 (프론트: React + TS + Vite)
│  ├─ src/              # 프론트엔드 (React)
│  └─ src-tauri/        # Rust 백엔드 + 설정
│     ├─ tauri.conf.json   # identifier: kr.doweb.stt
│     └─ gen/
│        ├─ apple/      # iOS Xcode 프로젝트 (자동생성)
│        └─ android/    # Android Gradle 프로젝트 (자동생성)
└─ whisper-live-kit/    # STT 백엔드 서버 (Python)
```

## 사전 준비물 (이 맥에는 모두 설치 완료)

- Node 26 / pnpm 10
- Rust(rustup) + 모바일 타겟(iOS/Android)
- Xcode 26.5 + CocoaPods (iOS)
- Android SDK + NDK 28 (`~/Library/Android/sdk`) — 환경변수는 `~/.zshrc`에 등록됨

> 새 터미널을 열면 `ANDROID_HOME` / `NDK_HOME` 가 자동 적용됩니다.

## 실행 (모두 `app/` 폴더에서)

```bash
cd app

# macOS 데스크톱
pnpm tauri dev
pnpm tauri build            # 릴리스 .app / .dmg

# iOS (시뮬레이터는 서명 불필요 / 실기기는 Apple 개발자 팀 필요)
pnpm tauri ios dev
pnpm tauri ios build

# Android (에뮬레이터 또는 USB 기기 필요 — Android Studio에서 AVD 생성)
pnpm tauri android dev
pnpm tauri android build
```

## Windows

macOS에서는 Windows 네이티브 빌드가 불가능합니다(MSVC 툴체인 필요). 프로젝트 자체는 Windows 호환이며, 빌드가 필요해지면:

- **GitHub Actions CI** (`tauri-action`)로 Windows/macOS/Linux를 한 번에 빌드하거나
- 실제 Windows PC에서 `pnpm install && pnpm tauri build` 실행

## STT 백엔드 (WhisperLiveKit) 실행

```bash
cd whisper-live-kit
pip install whisperlivekit         # 또는: uv sync
wlk --model base --language ko     # ws://localhost:8000 에서 대기
```

앱은 WebSocket(`ws://<서버>:8000/asr`)으로 오디오를 스트리밍해 실시간 전사 결과를 받습니다. (프론트엔드 연동은 다음 작업 단계)
