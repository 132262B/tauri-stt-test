"""온라인 화자 분리(임베딩 클러스터링) — docs/02-architecture.md I.

resemblyzer VoiceEncoder(d-vector, CPU)로 확정 세그먼트 임베딩을 뽑아, 코사인 임계값
기반으로 활성 화자 트랙(centroid)에 매칭/신규 생성한다. 청크 간 동일 화자 유지가 핵심.
모델 적재/임포트 실패 시 None 을 반환(전사는 영향 없음).
"""
import numpy as np


def _cos(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class OnlineDiarizer:
    def __init__(self, threshold: float = 0.62, min_sec: float = 0.4):
        from resemblyzer import VoiceEncoder

        self.enc = VoiceEncoder()
        self.threshold = threshold
        self.min_samples = int(16000 * min_sec)
        # 각 트랙: [centroid(np.ndarray), count]
        self.tracks: list[list] = []

    def assign(self, wav16k: np.ndarray):
        """16kHz mono 세그먼트 → 화자 트랙 id(int) 또는 None."""
        from resemblyzer import preprocess_wav

        if wav16k is None or len(wav16k) < self.min_samples:
            return None
        try:
            w = preprocess_wav(wav16k, source_sr=16000)
            if len(w) < self.min_samples:
                return None
            emb = self.enc.embed_utterance(w)
        except Exception:
            return None

        if not self.tracks:
            self.tracks.append([emb, 1])
            return 0
        sims = [_cos(emb, t[0]) for t in self.tracks]
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            c, n = self.tracks[best]
            self.tracks[best][0] = (c * n + emb) / (n + 1)
            self.tracks[best][1] = n + 1
            return best
        self.tracks.append([emb, 1])
        return len(self.tracks) - 1
