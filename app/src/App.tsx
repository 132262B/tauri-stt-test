import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";

// P0: 빈 레이아웃 셸. 좌(전사) / 우(컨트롤·자원 모니터) 2분할.
// 실제 전사·모니터·컨트롤은 P1 이후 components/ 로 채운다 (docs/02-architecture.md F).
function App() {
  const [ipc, setIpc] = useState("연결 확인 중…");
  const [capturing, setCapturing] = useState(false);
  const [capErr, setCapErr] = useState("");

  useEffect(() => {
    invoke<string>("ping")
      .then((r) => setIpc(`IPC ${r}`))
      .catch((e) => setIpc(`IPC 오류: ${e}`));
  }, []);

  async function toggleCapture() {
    setCapErr("");
    try {
      if (capturing) {
        await invoke("stop_capture");
        setCapturing(false);
      } else {
        await invoke("start_capture");
        setCapturing(true);
      }
    } catch (e) {
      setCapErr(String(e));
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
          <p className="placeholder">P1에서 실시간 화자 라벨 전사가 여기에 표시됩니다.</p>
        </section>
        <aside className="pane control-pane">
          <section className="panel">
            <h2>컨트롤</h2>
            <button onClick={toggleCapture}>
              {capturing ? "■ 마이크 정지" : "● 마이크 캡처 시작"}
            </button>
            <p className="placeholder">
              {capturing
                ? "캡처 중 — 16kHz mono RMS가 dev 콘솔에 출력됩니다."
                : "백엔드·입력 선택, 전사는 P1 후속 커밋."}
            </p>
            {capErr && <p className="error">{capErr}</p>}
          </section>
          <section className="panel">
            <h2>자원 모니터</h2>
            <p className="placeholder">CPU · 메모리 · 지연 · RTF (P1)</p>
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;
