"""사이드카 통신 프레이밍 (docs/02-architecture.md C.2).

- 제어(Rust→py) = stdin NDJSON, 결과(py→Rust) = stdout NDJSON, 로그 = stderr.
- PCM(Rust→py) = Unix Domain Socket: `u32 LE n_samples ‖ f32 LE * n ‖ f64 LE t_end`.
  Rust 가 bind(listen)하고 사이드카가 connect 하는 client 다.
"""
import json
import socket
import struct
import sys
import threading

import numpy as np


def write_msg(obj):
    """결과 1줄(NDJSON)을 stdout 으로. stdout 은 프레임 전용(라이브러리 print 오염 금지)."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def log(*args):
    print("[sidecar]", *args, file=sys.stderr, flush=True)


class PcmReceiver(threading.Thread):
    """Rust 가 bind 한 UDS 에 connect 해 PCM 프레임을 받아 on_pcm(samples, t_end) 호출."""

    def __init__(self, uds_path, on_pcm):
        super().__init__(daemon=True)
        self.uds_path = uds_path
        self.on_pcm = on_pcm
        self._stop = threading.Event()
        self._sock = None

    def run(self):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect(self.uds_path)
        except OSError as e:
            log(f"UDS connect 실패: {e}")
            return
        self._sock = sock
        with sock:
            while not self._stop.is_set():
                hdr = self._recvn(sock, 4)
                if hdr is None:
                    break
                (n,) = struct.unpack("<I", hdr)
                payload = self._recvn(sock, n * 4 + 8)
                if payload is None:
                    break
                samples = np.frombuffer(payload[: n * 4], dtype="<f4").astype(np.float32)
                (t_end,) = struct.unpack("<d", payload[n * 4 : n * 4 + 8])
                self.on_pcm(samples, t_end)

    def _recvn(self, sock, n):
        data = b""
        while len(data) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = sock.recv(n - len(data))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not chunk:
                return None
            data += chunk
        return data

    def stop(self):
        self._stop.set()
