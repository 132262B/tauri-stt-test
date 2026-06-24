"""ASRToken — whisper-live-kit timed_objects.ASRToken 의 경량 이식.

원본: whisper-live-kit/whisperlivekit/timed_objects.py (TimedText/ASRToken).
사이드카는 server 의존성 없이 토큰 단위 (start,end,text,probability)만 필요하다.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRToken:
    start: float
    end: float
    text: str
    probability: Optional[float] = None

    def with_offset(self, offset: float) -> "ASRToken":
        return ASRToken(self.start + offset, self.end + offset, self.text, self.probability)
