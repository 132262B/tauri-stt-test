"""stt_mlx — 온디바이스 MLX Whisper 사이드카.

Rust 백엔드(stt-asr-sidecar)가 spawn 하여 제어(stdin NDJSON)/결과(stdout NDJSON)/
PCM(UDS)로 통신한다. LocalAgreement·MLX 어댑터는 whisper-live-kit 에서 이식.
"""
