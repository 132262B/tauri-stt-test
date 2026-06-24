# stt-mlx-sidecar

온디바이스 MLX Whisper 사이드카 (Mac 전용). Rust 백엔드(`stt-asr-sidecar`)가 spawn 하여
**제어=stdin NDJSON / 결과=stdout NDJSON / PCM=Unix Domain Socket** 로 통신한다.
LocalAgreement-2 정책과 MLX Whisper 어댑터는 `whisper-live-kit` 에서 이식했다
(docs/02-architecture.md C·D, 결정 1·2).

> 모든 설치는 **프로젝트 내부** 에만 한다(전역 금지). venv·모델캐시 모두 이 폴더 안.

## 개발 환경 (프로젝트 로컬)

```sh
cd app/src-tauri/sidecar
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export HF_HOME="$PWD/.hf-cache"   # 모델 캐시도 프로젝트 내부
```

## 헤드리스 자가 검증

WAV 를 1초 청크로 스트리밍 시뮬레이션해 LocalAgreement 전사를 검증한다(프로토콜 불필요):

```sh
export HF_HOME="$PWD/.hf-cache"
./.venv/bin/python -m stt_mlx.main --self-test test-data/jfk.wav \
    --model mlx-community/whisper-large-v3-turbo
```

## 프로토콜 모드 (Rust 가 사용)

stdin 으로 NDJSON 제어, stdout 으로 NDJSON 결과:

- `{"type":"configure","model":"...","lang":null,"uds_path":"/tmp/...","trimming_sec":15}`
  → `{"type":"ready","backend":"mlx_whisper","model":"...","sr":16000}`
- `{"type":"process_iter"}` → `{"type":"tokens","committed":[{start,end,text,probability}],"buffer":"...","upto":f,"is_final":false}`
- `{"type":"set_language","lang":"ko"}` / `{"type":"change_speaker","at":12.3}`
- `{"type":"warmup"}` → `{"type":"warmed"}`
- `{"type":"finish"}` → 잔여 토큰(`is_final:true`) → `{"type":"bye"}`

PCM 은 `uds_path`(Rust 가 bind)에 connect 하여 프레임 수신:
`u32 LE n_samples ‖ f32 LE × n ‖ f64 LE t_end`.

## 파일

| 파일 | 역할 |
|---|---|
| `stt_mlx/timed.py` | `ASRToken` (wlk timed_objects 이식) |
| `stt_mlx/online_asr.py` | LocalAgreement-2 (`HypothesisBuffer`/`OnlineASRProcessor`, wlk 이식, segment 트리밍) |
| `stt_mlx/mlx_backend.py` | MLX Whisper 어댑터 (wlk MLXWhisper 이식) |
| `stt_mlx/framing.py` | NDJSON + UDS PCM 수신 |
| `stt_mlx/main.py` | 엔트리(프로토콜 루프 / `--self-test`) |
