"""사이드카 엔트리포인트.

기본: stdin NDJSON 제어 루프(configure/process_iter/set_language/change_speaker/
warmup/finish), 결과는 stdout NDJSON. PCM 은 UDS(PcmReceiver).
--self-test WAV: 프로토콜 없이 WAV 를 1초 청크로 스트리밍 시뮬레이션해 전사 출력(헤드리스 검증용).
"""
import argparse
import json
import sys
import threading

import numpy as np

from .framing import PcmReceiver, log, write_msg
from .mlx_backend import MLXWhisperBackend
from .online_asr import OnlineASRProcessor

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def tokens_to_json(tokens):
    return [
        {"start": t.start, "end": t.end, "text": t.text, "probability": t.probability}
        for t in tokens
    ]


class Session:
    def __init__(self, model_path, language, trimming_sec=15.0):
        self.backend = MLXWhisperBackend(model_path, language)
        self.proc = OnlineASRProcessor(self.backend, buffer_trimming_sec=trimming_sec)
        self.lock = threading.Lock()
        self.receiver = None

    def attach_uds(self, uds_path):
        self.receiver = PcmReceiver(uds_path, self._on_pcm)
        self.receiver.start()

    def _on_pcm(self, samples, _t_end):
        with self.lock:
            self.proc.insert_audio_chunk(samples)

    def process_iter(self):
        with self.lock:
            committed = self.proc.process_iter()
            return committed, self.proc.get_buffer(), self.proc.get_audio_buffer_end_time()

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
                session = Session(model, msg.get("lang"), float(msg.get("trimming_sec", 15.0)))
                if msg.get("uds_path"):
                    session.attach_uds(msg["uds_path"])
                write_msg({"type": "ready", "backend": "mlx_whisper", "model": model, "sr": 16000})
            elif mtype == "process_iter":
                if session is None:
                    write_msg({"type": "error", "code": "not_configured", "msg": "configure first"})
                    continue
                committed, buffer_text, upto = session.process_iter()
                write_msg({"type": "tokens", "committed": tokens_to_json(committed),
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
                    session.backend.transcribe(np.zeros(16000, dtype=np.float32))
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
    backend = MLXWhisperBackend(model, lang)
    proc = OnlineASRProcessor(backend, buffer_trimming_sec=15.0)
    chunk = 16000  # 1초
    committed_all = []
    for i in range(0, len(audio), chunk):
        proc.insert_audio_chunk(audio[i : i + chunk])
        committed = proc.process_iter()
        committed_all += committed
        if committed:
            log("COMMIT:", "".join(t.text for t in committed))
        buf = proc.get_buffer()
        if buf:
            log("  buffer:", buf)
    committed_all += proc.finish()
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
