import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";

// P1 커밋9: 마이크 → 사이드카(MLX Whisper) → transcript_update 수신·렌더.
// 화자 라벨/자원 모니터/백엔드 전환은 P1.5/P2.
interface TranscriptSnapshot {
  committedText: string;
  buffer: string;
  upto: number;
}

interface Metrics {
  cpuPct: number;
  rssMb: number;
  sidecarRssMb: number;
  rtf: number;
  latencyMsP50: number;
  latencyMsP95: number;
  backend: string;
  model: string;
}

function App() {
  const [ipc, setIpc] = useState("연결 확인 중…");
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState("");
  const [committed, setCommitted] = useState("");
  const [buffer, setBuffer] = useState("");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    invoke<string>("ping")
      .then((r) => setIpc(`IPC ${r}`))
      .catch((e) => setIpc(`IPC 오류: ${e}`));

    const un1 = listen<TranscriptSnapshot>("transcript_update", (e) => {
      setCommitted(e.payload.committedText);
      setBuffer(e.payload.buffer);
    });
    const un2 = listen("transcript_done", () => setBuffer(""));
    const un3 = listen<Metrics>("metrics_update", (e) => setMetrics(e.payload));
    return () => {
      un1.then((f) => f());
      un2.then((f) => f());
      un3.then((f) => f());
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [committed, buffer]);

  async function toggle() {
    setErr("");
    try {
      if (running) {
        await invoke("stop_session");
        setRunning(false);
      } else {
        setCommitted("");
        setBuffer("");
        await invoke("start_session");
        setRunning(true);
      }
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>온디바이스 회의 전사</h1>
        <span className="ipc-status">{ipc}</span>
      </header>
      <main className="app-body">
        <section className="pane transcript-pane">
          <h2>전사</h2>
          {!committed && !buffer && (
            <p className="placeholder">
              {running ? "듣는 중… 말하면 전사가 나타납니다." : "마이크 캡처를 시작하세요."}
            </p>
          )}
          <p className="transcript">
            <span className="committed">{committed}</span>
            {buffer && <span className="partial"> {buffer}</span>}
          </p>
          <div ref={bottomRef} />
        </section>
        <aside className="pane control-pane">
          <section className="panel">
            <h2>컨트롤</h2>
            <button onClick={toggle} className={running ? "stop" : "start"}>
              {running ? "■ 전사 정지" : "● 전사 시작 (마이크)"}
            </button>
            <p className="placeholder">
              {running
                ? "MLX Whisper(turbo) 온디바이스 전사 중. 첫 시작은 모델 로딩에 수 초."
                : "백엔드 전환·자원 모니터는 다음 단계(P1.5/P2)."}
            </p>
            {err && <p className="error">{err}</p>}
          </section>
          <section className="panel">
            <h2>자원 모니터</h2>
            {metrics ? (
              <div className="metrics">
                <div className="metric">
                  <span>메모리 (앱)</span>
                  <b>{metrics.rssMb.toFixed(0)} MB</b>
                </div>
                <div className="metric">
                  <span>메모리 (사이드카/MLX)</span>
                  <b>{metrics.sidecarRssMb.toFixed(0)} MB</b>
                </div>
                <div className="metric">
                  <span>CPU</span>
                  <b>{metrics.cpuPct.toFixed(0)} %</b>
                </div>
                <div className="metric">
                  <span>RTF</span>
                  <b className={metrics.rtf > 1 ? "warn" : ""}>{metrics.rtf.toFixed(2)}</b>
                </div>
                <div className="metric">
                  <span>추론 p50/p95</span>
                  <b>
                    {metrics.latencyMsP50.toFixed(0)}/{metrics.latencyMsP95.toFixed(0)} ms
                  </b>
                </div>
              </div>
            ) : (
              <p className="placeholder">전사 시작 시 1초마다 갱신됩니다.</p>
            )}
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;
