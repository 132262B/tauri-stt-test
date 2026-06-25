"""Voxtral / Qwen3-ASR 백엔드 + 자체 스트리밍 래퍼.

Voxtral(mlx-voxtral)·Qwen3-ASR(qwen3-asr-mlx)는 버퍼 전체를 재전사하는 "자체 스트리밍"
방식이라 Whisper 의 LocalAgreement(단어 타임스탬프 기반)와 다르다. 여기서는 단어 prefix-commit
(연속 2회 전사의 최장 공통 접두사를 확정)으로 OnlineASRProcessor 와 동일한 인터페이스
(insert_audio_chunk/process_iter/get_buffer/get_audio_buffer_end_time/finish)를 노출한다.
타임스탬프는 윈도우 길이에 비례한 근사값(단어 단위 ts 미제공).
"""
import os
import tempfile

import numpy as np
import soundfile as sf

from .timed import ASRToken


def pick_backend(model_id: str) -> str:
    m = (model_id or "").lower()
    if "voxtral" in m:
        return "voxtral"
    if "qwen" in m:
        return "qwen"
    return "whisper"


class VoxtralFullBackend:
    sep = " "

    def __init__(self, model_id, language=None):
        from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

        self.model = VoxtralForConditionalGeneration.from_pretrained(model_id)
        self.processor = VoxtralProcessor.from_pretrained(model_id)
        self.language = language or "en"
        self._tmp = os.path.join(tempfile.gettempdir(), f"voxtral_buf_{os.getpid()}.wav")

    def set_language(self, lang):
        self.language = lang or "en"

    def transcribe_full(self, audio_np) -> str:
        sf.write(self._tmp, audio_np, 16000)
        inputs = self.processor.apply_transcrition_request(language=self.language, audio=self._tmp)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
        return self.processor.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()


class QwenFullBackend:
    sep = " "

    def __init__(self, model_id, language=None):
        from qwen3_asr_mlx import Qwen3ASR

        self.model = Qwen3ASR.from_pretrained(model_id)
        self.language = language
        self._tmp = os.path.join(tempfile.gettempdir(), f"qwen_buf_{os.getpid()}.wav")

    def set_language(self, lang):
        self.language = lang

    def transcribe_full(self, audio_np) -> str:
        sf.write(self._tmp, audio_np, 16000)
        return self.model.transcribe(self._tmp).text.strip()


class SelfStreamingProcessor:
    """버퍼 전체 재전사 + 단어 prefix-commit. OnlineASRProcessor 와 동일 인터페이스."""

    SAMPLING_RATE = 16000

    def __init__(self, backend, window_sec: float = 24.0):
        self.backend = backend
        self.window = int(self.SAMPLING_RATE * window_sec)
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0.0
        self.committed_n = 0
        self.last_words: list[str] = []

    def insert_audio_chunk(self, samples):
        self.audio = np.append(self.audio, samples)

    def get_audio_buffer_end_time(self) -> float:
        return self.offset + len(self.audio) / self.SAMPLING_RATE

    def _tokens(self, words, lo, hi, total):
        dur = len(self.audio) / self.SAMPLING_RATE
        out = []
        for j in range(lo, hi):
            s = self.offset + (j / max(total, 1)) * dur
            e = self.offset + ((j + 1) / max(total, 1)) * dur
            out.append(ASRToken(s, e, (" " if j > 0 else "") + words[j]))
        return out

    def process_iter(self):
        if len(self.audio) < int(self.SAMPLING_RATE * 0.5):
            return []
        words = self.backend.transcribe_full(self.audio).split()
        lcp = 0
        while lcp < len(words) and lcp < len(self.last_words) and words[lcp] == self.last_words[lcp]:
            lcp += 1
        committed = []
        if lcp > self.committed_n:
            committed = self._tokens(words, self.committed_n, lcp, len(words))
            self.committed_n = lcp
        self.last_words = words

        # 윈도우 초과 시 flush(전부 확정 후 초기화)
        if len(self.audio) >= self.window:
            if len(words) > self.committed_n:
                committed += self._tokens(words, self.committed_n, len(words), len(words))
            self.offset = self.get_audio_buffer_end_time()
            self.audio = np.zeros(0, dtype=np.float32)
            self.committed_n = 0
            self.last_words = []
        return committed

    def get_buffer(self) -> str:
        return " ".join(self.last_words[self.committed_n :])

    def finish(self):
        if len(self.last_words) > self.committed_n:
            return self._tokens(self.last_words, self.committed_n, len(self.last_words), len(self.last_words))
        return []
