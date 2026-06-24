"""LocalAgreement-2 스트리밍 정책 — whisper-live-kit online_asr.py 의 경량 이식.

원본: whisper-live-kit/whisperlivekit/local_agreement/online_asr.py
(HypothesisBuffer + OnlineASRProcessor). 서버/문장토크나이저 의존성을 제거하고
**segment 트리밍 경로만** 남겼다(한국어는 문장 토크나이저 미지원 → segment 강제,
docs/02-architecture.md 2.3·7장). sep 은 백엔드 제공(MLX Whisper="").
"""
import sys
from typing import List, Optional, Tuple

import numpy as np

from .timed import ASRToken


class HypothesisBuffer:
    """확정(committed)/미확정(buffer)/신규(new) 토큰 버퍼.

    연속 2회 추론에서 동일한 최장 공통 접두사만 commit (LocalAgreement-2).
    원본 online_asr.py:11-94 의 1:1 이식.
    """

    def __init__(self, confidence_validation=False):
        self.confidence_validation = confidence_validation
        self.committed_in_buffer: List[ASRToken] = []
        self.buffer: List[ASRToken] = []
        self.new: List[ASRToken] = []
        self.last_committed_time = 0.0
        self.last_committed_word: Optional[str] = None

    def insert(self, new_tokens: List[ASRToken], offset: float):
        new_tokens = [token.with_offset(offset) for token in new_tokens]
        self.new = [t for t in new_tokens if t.start > self.last_committed_time - 0.1]
        if self.new:
            first = self.new[0]
            if abs(first.start - self.last_committed_time) < 1:
                if self.committed_in_buffer:
                    committed_len = len(self.committed_in_buffer)
                    new_len = len(self.new)
                    max_ngram = min(min(committed_len, new_len), 5)
                    for i in range(1, max_ngram + 1):
                        committed_ngram = " ".join(t.text for t in self.committed_in_buffer[-i:])
                        new_ngram = " ".join(t.text for t in self.new[:i])
                        if committed_ngram == new_ngram:
                            for _ in range(i):
                                self.new.pop(0)
                            break

    def flush(self) -> List[ASRToken]:
        committed: List[ASRToken] = []
        while self.new:
            current_new = self.new[0]
            if self.confidence_validation and current_new.probability and current_new.probability > 0.95:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.new.pop(0)
                if self.buffer:
                    self.buffer.pop(0)
            elif not self.buffer:
                break
            elif current_new.text == self.buffer[0].text:
                committed.append(current_new)
                self.last_committed_word = current_new.text
                self.last_committed_time = current_new.end
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(committed)
        return committed

    def pop_committed(self, time: float):
        while self.committed_in_buffer and self.committed_in_buffer[0].end <= time:
            self.committed_in_buffer.pop(0)


class OnlineASRProcessor:
    """스트리밍 오디오를 누적해 주기적으로 ASR 을 호출하고 LocalAgreement 로 확정/트림.

    원본 OnlineASRProcessor (segment 트리밍 경로). asr 는 transcribe/ts_words/
    segments_end_ts/sep 를 제공해야 한다(mlx_backend.MLXWhisperBackend).
    """
    SAMPLING_RATE = 16000

    def __init__(self, asr, buffer_trimming_sec: float = 15.0):
        self.asr = asr
        self.buffer_trimming_sec = buffer_trimming_sec
        self.init()

    def init(self, offset: Optional[float] = None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = offset if offset is not None else 0.0
        self.transcript_buffer.last_committed_time = self.buffer_time_offset
        self.committed: List[ASRToken] = []
        self.time_of_last_asr_output = 0.0

    def get_audio_buffer_end_time(self) -> float:
        return self.buffer_time_offset + (len(self.audio_buffer) / self.SAMPLING_RATE)

    def insert_audio_chunk(self, audio: np.ndarray):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def new_speaker(self, at: float):
        """화자 전환 시 컨텍스트 리셋(docs/02-architecture.md D.2 on_change_speaker)."""
        committed = self.process_iter()
        self.init(offset=at)
        return committed

    def prompt(self) -> str:
        """버퍼 밖 확정 텍스트의 끝 200자를 다음 추론 프롬프트로(원본 :187-209)."""
        k = len(self.committed)
        while k > 0 and self.committed[k - 1].end > self.buffer_time_offset:
            k -= 1
        prompt_words = [t.text for t in self.committed[:k]]
        prompt_list = []
        length_count = 0
        while prompt_words and length_count < 200:
            word = prompt_words.pop(-1)
            length_count += len(word) + 1
            prompt_list.append(word)
        return self.asr.sep.join(prompt_list[::-1])

    def get_buffer(self) -> str:
        return self.asr.sep.join(t.text for t in self.transcript_buffer.buffer)

    def process_iter(self) -> List[ASRToken]:
        """현재 오디오 버퍼를 추론하고 확정 토큰 리스트를 반환."""
        prompt_text = self.prompt()
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt_text)
        tokens = self.asr.ts_words(res)
        self.transcript_buffer.insert(tokens, self.buffer_time_offset)
        committed_tokens = self.transcript_buffer.flush()
        self.committed.extend(committed_tokens)
        if committed_tokens:
            self.time_of_last_asr_output = self.committed[-1].end

        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        # 무출력 freeze 방지(원본 :244-252)
        if not committed_tokens and buffer_duration > self.buffer_trimming_sec:
            if self.get_audio_buffer_end_time() - self.time_of_last_asr_output > self.buffer_trimming_sec:
                self.init(offset=self.get_audio_buffer_end_time())
                return []
        # segment 트리밍(한국어 강제)
        if buffer_duration > self.buffer_trimming_sec:
            self.chunk_completed_segment(res)
        return committed_tokens

    def chunk_completed_segment(self, res):
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if not self.committed:
            if buffer_duration > self.buffer_trimming_sec:
                self.chunk_at(self.buffer_time_offset + buffer_duration / 2)
            return
        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.committed[-1].end
        chunk_done = False
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > last_committed_time:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= last_committed_time:
                self.chunk_at(e)
                chunk_done = True
        if not chunk_done and buffer_duration > self.buffer_trimming_sec:
            self.chunk_at(last_committed_time)

    def chunk_at(self, time: float):
        self.transcript_buffer.pop_committed(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def finish(self) -> List[ASRToken]:
        remaining = self.transcript_buffer.buffer
        self.buffer_time_offset = self.get_audio_buffer_end_time()
        return remaining
