"""MLX Whisper ASR 백엔드 — whisper-live-kit backends.MLXWhisper 의 경량 이식.

원본: whisper-live-kit/whisperlivekit/local_agreement/backends.py:157-216.
mlx_whisper.transcribe(word_timestamps=True) 결과(segments[].words[])에서
단어 타임스탬프 토큰을 추출한다. Apple Silicon MLX 온디바이스 추론.
"""
from typing import List, Optional

from .timed import ASRToken


class MLXWhisperBackend:
    sep = ""  # 토큰 결합 구분자 (MLX Whisper 는 "" — 원본 backends.py:160)

    def __init__(self, model_path: str, language: Optional[str] = None):
        """model_path: HF repo id (예: 'mlx-community/whisper-large-v3-turbo').
        language: None=자동감지(한·영 코드스위칭)."""
        self.model_path = model_path
        self.language = language
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder, transcribe

        # 모델을 미리 적재(첫 추론 지연 제거)
        ModelHolder.get_model(model_path, mx.float16)
        self._transcribe = transcribe

    def set_language(self, language: Optional[str]):
        self.language = language

    def transcribe(self, audio, init_prompt: str = ""):
        res = self._transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            language=self.language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        return res.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens: List[ASRToken] = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                tokens.append(ASRToken(word["start"], word["end"], word["word"]))
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [s["end"] for s in segments]
