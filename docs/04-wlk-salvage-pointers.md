# WhisperLiveKit 살베지 포인터 — 참고본 삭제 후 위치 기록

참고본 `whisper-live-kit/`(Python 원본)을 저장소에서 제거했다. 이 문서는 **아직 Rust 로
이식하지 않은** 알고리즘이 원본의 어디에 있었는지 포인터만 남긴다. 필요할 때 복구 경로:

- 공개 upstream: <https://github.com/QuentinFuxa/WhisperLiveKit>
- 이 저장소 git 히스토리: `git log --oneline -- whisper-live-kit` → 해당 커밋에서 `git show`
- 설계 분석: `docs/01-whisper-live-kit-analysis.md` (서브시스템별 file:line 근거 보존)

## 이미 이식됨(코드에 존재 — 원본 불필요)

| 알고리즘 | 원본 위치 | 이식처 |
|---|---|---|
| WER(S/I/D)·타임스탬프 정확도 | `whisperlivekit/metrics.py:12-156` | `crates/asr-core/src/eval.rs` |
| CJK 분할·LIS 단조보정·글자수 가중 | `qwen3_vllm_asr.py:160-263` | `crates/asr-core/src/asr/wordtime.rs` |
| 인코딩 파일 디코드(능력) | `ffmpeg_manager.py`(FFmpeg) | `src/capture/file_src.rs`(Symphonia, in-process) |

## 미이식(필요 시 upstream/히스토리에서 참고)

| 알고리즘 | 원본 위치 | 메모 |
|---|---|---|
| ForcedAligner(실제 워드 타임스탬프, 2nd 모델) | `qwen3_vllm_asr.py:307-322,354-408` | 현 antirez/qwen-asr C 경로는 평문만 반환 → 별도 모델 필요. wordtime.rs 가 빌딩블록 |
| SimulStreaming/AlignAtt | `simul_whisper/align_att_base.py:272-279(frame_threshold)·253-268(rewind)·485-530(DRY)·436-481(UTF-8 pending)`, `simul_whisper/mlx/simul_whisper.py:351-401(cross-attn)` | 저지연 토큰 스트리밍을 재검토할 때. UTF-8 pending 은 한국어 토큰레벨 스트리밍 시 필수 |
| VAC 무음 회계 글루 | `audio_processor.py:148-194·781-819·798-810·382-397` | 무음 제거+드리프트 없는 타임스탬프 클램핑. Silero VAD 도입 시 |
| tokens_alignment 라인빌더·300s prune | `tokens_alignment.py:97-127·179-209·57-83` | 라이브 화자라인·문장단위 배정·장시간 보존 한계 |
| Sortformer 스트리밍 화자분리 | `diarization/sortformer_backend.py` | 라이브 화자 라벨링을 재검토할 때(spkcache/fifo/preds) |
| HF 벤치마크 데이터셋 카탈로그 | `benchmark/datasets.py:66-208` | 표준 다국어 WER 코퍼스가 필요해질 때 |
