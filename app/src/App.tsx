import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { save } from "@tauri-apps/plugin-dialog";
import "./App.css";

// P1.5: 마이크 → 사이드카(MLX Whisper + 화자분리) → 화자별 라인 렌더 + 자원 모니터 + 내보내기.
interface TranscriptLine {
  speaker: number | null;
  text: string;
  start: number;
  end: number;
}
interface TranscriptSnapshot {
  committedText: string;
  lines: TranscriptLine[];
  buffer: string;
  bufferSpeaker: number | null;
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

// 교체 가능한 ASR 백엔드/모델. 라벨에 엔진명을 명시(테스트 앱: 백엔드별 속도/메모리 비교).
// 전부 Rust 네이티브(whisper.cpp/Metal, in-process). Python/Node 런타임 0.
const MODELS: { id: string; label: string }[] = [
  { id: "ggml-base", label: "Whisper · base (74M · 권장·빠름)" },
  { id: "ggml-small", label: "Whisper · small (244M)" },
  { id: "sensevoice", label: "SenseVoice · 다국어(한·영·일·중)" },
  { id: "ggml-tiny", label: "Whisper · tiny (39M · 가장 빠름)" },
  { id: "ggml-large-v3-turbo", label: "Whisper · turbo (809M · ⚠️ 현재 로드 실패 가능)" },
  { id: "ggml-large-v3", label: "Whisper · large-v3 (1.55B · ⚠️ 로드 실패 가능)" },
];

const SPEAKER_COLORS = ["#2e7d32", "#1565c0", "#c2185b", "#e67e22", "#6a1b9a", "#00838f"];
function speakerColor(s: number | null): string {
  return s == null ? "#888" : SPEAKER_COLORS[s % SPEAKER_COLORS.length];
}
function speakerName(s: number | null): string {
  return s == null ? "화자?" : `화자 ${s + 1}`;
}

function App() {
  const [ipc, setIpc] = useState("연결 확인 중…");
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState("");
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [buffer, setBuffer] = useState("");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [model, setModel] = useState(MODELS[0].id);
  const [lang, setLang] = useState(""); // "" = 자동
  const [input, setInput] = useState("mic"); // mic | system | both
  const [devices, setDevices] = useState<string[]>([]);
  const [device, setDevice] = useState(""); // "" = 기본 장치
  const [level, setLevel] = useState(0); // 입력 RMS (0..~0.3)
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    invoke<string>("ping")
      .then((r) => setIpc(`IPC ${r}`))
      .catch((e) => setIpc(`IPC 오류: ${e}`));

    invoke<string[]>("list_inputs")
      .then((ds) => {
        setDevices(ds);
        // 내장 마이크 자동 우선 선택(iPhone/BlackHole 등 무음 장치 회피)
        const builtin = ds.find((d) => /MacBook|내장|Built-?in/i.test(d));
        if (builtin) setDevice(builtin);
      })
      .catch(() => {});

    const un1 = listen<TranscriptSnapshot>("transcript_update", (e) => {
      setLines(e.payload.lines);
      setBuffer(e.payload.buffer);
    });
    const un2 = listen("transcript_done", () => setBuffer(""));
    const un3 = listen<Metrics>("metrics_update", (e) => setMetrics(e.payload));
    const un4 = listen<number>("audio_level", (e) => setLevel(e.payload));
    return () => {
      un1.then((f) => f());
      un2.then((f) => f());
      un3.then((f) => f());
      un4.then((f) => f());
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines, buffer]);

  async function exportAs(fmt: "txt" | "srt" | "json") {
    setErr("");
    try {
      const path = await save({
        defaultPath: `transcript.${fmt}`,
        filters: [{ name: fmt.toUpperCase(), extensions: [fmt] }],
      });
      if (!path) return;
      await invoke("export_transcript", { path, format: fmt });
    } catch (e) {
      setErr(String(e));
    }
  }

  // macOS 권한 확인/요청. 문제 시 안내 문자열 반환, 정상이면 null. 비-macOS는 통과.
  async function ensurePermissions(): Promise<string | null> {
    const need = async (check: string, request: string) => {
      let ok = await invoke<boolean>(`plugin:macos-permissions|${check}`);
      if (!ok) {
        await invoke(`plugin:macos-permissions|${request}`);
        ok = await invoke<boolean>(`plugin:macos-permissions|${check}`);
      }
      return ok;
    };
    try {
      if (input === "mic" || input === "both") {
        if (!(await need("check_microphone_permission", "request_microphone_permission"))) {
          return "🎤 마이크 권한이 없습니다. 시스템 설정 → 개인정보 보호 및 보안 → 마이크 에서 이 앱을 허용한 뒤 다시 시도하세요.";
        }
      }
      if (input === "system" || input === "both") {
        if (!(await need("check_screen_recording_permission", "request_screen_recording_permission"))) {
          return "🖥 화면 녹화(시스템 오디오) 권한이 없습니다. 시스템 설정 → 개인정보 보호 및 보안 → 화면 및 시스템 오디오 녹화 에서 허용 후 앱을 재시작하세요.";
        }
      }
      return null;
    } catch {
      return null; // 플러그인 미존재(비-macOS) → 통과
    }
  }

  async function toggle() {
    setErr("");
    try {
      if (running) {
        await invoke("stop_session");
        setRunning(false);
        setLevel(0);
      } else {
        const permErr = await ensurePermissions();
        if (permErr) {
          setErr(permErr);
          return;
        }
        setLines([]);
        setBuffer("");
        await invoke("start_session", { model, lang, input, device });
        setRunning(true);
      }
    } catch (e) {
      setErr(String(e));
    }
  }

  const hasContent = lines.length > 0 || buffer.length > 0;

  return (
    <div className="app">
      <header className="app-header">
        <h1>온디바이스 회의 전사</h1>
        <span className="ipc-status">{ipc}</span>
      </header>
      <main className="app-body">
        <section className="pane transcript-pane">
          <h2>전사</h2>
          {!hasContent && (
            <p className="placeholder">
              {running ? "듣는 중… 말하면 화자별 전사가 나타납니다." : "마이크 캡처를 시작하세요."}
            </p>
          )}
          <div className="transcript">
            {lines.map((l, i) => (
              <div className="line" key={i}>
                {l.speaker != null && (
                  <span className="speaker-badge" style={{ background: speakerColor(l.speaker) }}>
                    {speakerName(l.speaker)}
                  </span>
                )}
                <span className="line-text">{l.text}</span>
              </div>
            ))}
            {buffer && (
              <div className="line partial-line">
                <span className="speaker-badge ghost">…</span>
                <span className="line-text partial">{buffer}</span>
              </div>
            )}
          </div>
          {lines.length > 0 && (
            <div className="export-bar">
              <span>내보내기:</span>
              <button onClick={() => exportAs("txt")}>txt</button>
              <button onClick={() => exportAs("srt")}>srt</button>
              <button onClick={() => exportAs("json")}>json</button>
            </div>
          )}
          <div ref={bottomRef} />
        </section>
        <aside className="pane control-pane">
          <section className="panel">
            <h2>컨트롤</h2>
            <label className="field">
              <span>모델</span>
              <select value={model} onChange={(e) => setModel(e.target.value)} disabled={running}>
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>언어</span>
              <select value={lang} onChange={(e) => setLang(e.target.value)} disabled={running}>
                <option value="">자동 감지 (한·영 혼용)</option>
                <option value="ko">한국어 고정</option>
                <option value="en">영어 고정</option>
              </select>
            </label>
            <label className="field">
              <span>입력 소스</span>
              <select value={input} onChange={(e) => setInput(e.target.value)} disabled={running}>
                <option value="mic">마이크</option>
                <option value="system">시스템 오디오 (회의 소리·화면녹화 권한)</option>
                <option value="both">마이크 + 시스템 오디오</option>
              </select>
            </label>
            {input !== "system" && (
              <label className="field">
                <span>입력 장치 (마이크 / BlackHole 등 출력캡처)</span>
                <select value={device} onChange={(e) => setDevice(e.target.value)} disabled={running}>
                  <option value="">기본 장치</option>
                  {devices.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <button onClick={toggle} className={running ? "stop" : "start"}>
              {running ? "■ 전사 정지" : "● 전사 시작"}
            </button>

            <div className="soundbar" title="입력 레벨">
              {Array.from({ length: 24 }).map((_, i) => {
                const frac = Math.min(1, Math.sqrt(level) * 2.4);
                const on = i < Math.round(frac * 24);
                const color = i < 15 ? "#2e7d32" : i < 21 ? "#e6a700" : "#c0392b";
                return (
                  <span key={i} className="seg" style={on ? { background: color } : undefined} />
                );
              })}
            </div>
            <p className={running ? (level > 0.002 ? "vu-on" : "vu-off") : "placeholder"}>
              {running
                ? `${level > 0.002 ? "🎙 입력 감지" : "🔇 거의 무음"} · 레벨 ${(level * 1000).toFixed(1)} (말하면 올라가야 정상; 계속 1 이하면 마이크 권한/입력 볼륨 확인)`
                : "전사 시작을 누르면 입력 레벨 막대가 움직입니다."}
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
