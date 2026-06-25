"""사이드카 엔트리포인트.

기본: stdin NDJSON 제어 루프(configure/process_iter/set_language/change_speaker/
warmup/finish), 결과는 stdout NDJSON. PCM 은 UDS(PcmReceiver).
--self-test WAV: 프로토콜 없이 WAV 를 1초 청크로 스트리밍 시뮬레이션해 전사+화자 출력(헤드리스 검증용).
"""
import argparse
import json
import sys
import threading

import numpy as np

from .framing import PcmReceiver, log, write_msg
from .mlx_backend import MLXWhisperBackend
from .online_asr import OnlineASRProcessor
from .streaming_backends import (
    QwenFullBackend,
    SelfStreamingProcessor,
    VoxtralFullBackend,
    pick_backend,
)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def tokens_to_json(tokens, speaker=None):
    return [
        {
            "start": t.start,
            "end": t.end,
            "text": t.text,
            "probability": t.probability,
            "speaker": speaker,
        }
        for t in tokens
    ]


class Session:
    def __init__(self, model_path, language, trimming_sec=15.0, diarize=True):
        self.kind = pick_backend(model_path)
        if self.kind == "voxtral":
            self.backend = VoxtralFullBackend(model_path, language)
            self.proc = SelfStreamingProcessor(self.backend)
        elif self.kind == "qwen":
            self.backend = QwenFullBackend(model_path, language)
            self.proc = SelfStreamingProcessor(self.backend)
        else:
            self.backend = MLXWhisperBackend(model_path, language)
            self.proc = OnlineASRProcessor(self.backend, buffer_trimming_sec=trimming_sec)
        log(f"backend={self.kind} model={model_path}")
        self.lock = threading.Lock()
        self.receiver = None
        self.full_audio = np.zeros(0, dtype=np.float32)
        self.diarizer = None
        if diarize:
            try:
                from .diarizer import OnlineDiarizer

                self.diarizer = OnlineDiarizer()
                log("diarizer 활성")
            except Exception as e:  # noqa: BLE001
                log("diarizer 비활성(임포트 실패):", e)

    def warmup(self):
        if self.kind == "whisper":
            self.backend.transcribe(np.zeros(16000, dtype=np.float32))
        # 자체 스트리밍 백엔드는 모델이 __init__ 에서 적재됨(첫 전사로 워밍)

    def attach_uds(self, uds_path):
        self.receiver = PcmReceiver(uds_path, lambda s, _t: self.feed(s))
        self.receiver.start()

    def feed(self, samples):
        with self.lock:
            self.proc.insert_audio_chunk(samples)
            self.full_audio = np.append(self.full_audio, samples)

    def _slice(self, start, end):
        a = int(max(0.0, start) * 16000)
        b = int(max(0.0, end) * 16000)
        if b <= a or a >= len(self.full_audio):
            return None
        return self.full_audio[a : min(b, len(self.full_audio))]

    def process_iter(self):
        with self.lock:
            committed = self.proc.process_iter()
            buffer_text = self.proc.get_buffer()
            upto = self.proc.get_audio_buffer_end_time()
            seg = self._slice(committed[0].start, committed[-1].end) if committed else None
        speaker = None
        if committed and self.diarizer is not None and seg is not None:
            speaker = self.diarizer.assign(seg)
        return committed, buffer_text, upto, speaker

    def change_speaker(self, at):
        with self.lock:
            return self.proc.new_speaker(at)

    def finish(self):
        with self.lock:
            return self.proc.finish()

    def close(self):
        if self.receiver:
            self.receiver.stop()


def run_protocol():
    session = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            write_msg({"type": "error", "code": "bad_json", "msg": str(e)})
            continue
        mtype = msg.get("type")
        try:
            if mtype == "configure":
                model = msg.get("model") or DEFAULT_MODEL
                session = Session(
                    model,
                    msg.get("lang"),
                    float(msg.get("trimming_sec", 15.0)),
                    diarize=bool(msg.get("diarize", True)),
                )
                if msg.get("uds_path"):
                    session.attach_uds(msg["uds_path"])
                write_msg({"type": "ready", "backend": "mlx_whisper", "model": model, "sr": 16000})
            elif mtype == "process_iter":
                if session is None:
                    write_msg({"type": "error", "code": "not_configured", "msg": "configure first"})
                    continue
                committed, buffer_text, upto, speaker = session.process_iter()
                write_msg({"type": "tokens", "committed": tokens_to_json(committed, speaker),
                           "buffer": buffer_text, "upto": upto, "is_final": False})
            elif mtype == "set_language":
                if session:
                    session.backend.set_language(msg.get("lang"))
            elif mtype == "change_speaker":
                if session:
                    committed = session.change_speaker(float(msg.get("at", 0.0)))
                    write_msg({"type": "tokens", "committed": tokens_to_json(committed),
                               "buffer": "", "upto": 0.0, "is_final": False})
            elif mtype == "warmup":
                if session:
                    session.warmup()
                    write_msg({"type": "warmed"})
            elif mtype == "finish":
                if session:
                    remaining = session.finish()
                    write_msg({"type": "tokens", "committed": tokens_to_json(remaining),
                               "buffer": "", "upto": 0.0, "is_final": True})
                    session.close()
                write_msg({"type": "bye"})
                break
            else:
                write_msg({"type": "error", "code": "unknown_type", "msg": str(mtype)})
        except Exception as e:  # noqa: BLE001
            import traceback
            log(traceback.format_exc())
            write_msg({"type": "error", "code": "exception", "msg": str(e)})


def _resample_linear(audio, sr, target=16000):
    if sr == target:
        return audio.astype(np.float32)
    n_out = int(len(audio) * target / sr)
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def run_self_test(path, model, lang):
    import soundfile as sf

    log(f"self-test: {path} (model={model})")
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = _resample_linear(audio, sr)
    sess = Session(model, lang, diarize=True)
    chunk = 16000  # 1초
    committed_all = []
    for i in range(0, len(audio), chunk):
        sess.feed(audio[i : i + chunk])
        committed, _buf, _upto, spk = sess.process_iter()
        committed_all += committed
        if committed:
            log(f"COMMIT[spk={spk}]:", "".join(t.text for t in committed))
    committed_all += sess.finish()
    full = "".join(t.text for t in committed_all)
    log("=== FINAL TRANSCRIPT ===")
    print(full)
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", dest="self_test", default=None, help="WAV 경로(헤드리스 검증)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lang", default=None)
    args = p.parse_args()
    if args.self_test:
        run_self_test(args.self_test, args.model, args.lang)
    else:
        run_protocol()


if __name__ == "__main__":
    main()
