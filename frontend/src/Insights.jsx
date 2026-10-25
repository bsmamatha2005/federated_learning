import { useEffect, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/api";

const ALGOS = ["FedAvg", "FedProx", "FedCurv"];
const ALGO_TEXT = {
  FedAvg: "Averages the hospitals' weights, weighted by how many patients each has.",
  FedProx: "FedAvg plus a penalty that stops a hospital's model drifting far from the global model. Helps when hospitals differ.",
  FedCurv: "Adds a Fisher-information penalty that protects weights other hospitals depend on, reducing forgetting.",
};
const ALGO_COLOR = { FedAvg: "#2563eb", FedProx: "#f97316", FedCurv: "#16a34a" };

const PHASES = [
  { title: "1. Send", color: "#2563eb",
    text: "The server sends the current global model to every hospital." },
  { title: "2. Train locally", color: "#f97316",
    text: "Each hospital trains the model on its own patients. Records never leave the building." },
  { title: "3. Return weights", color: "#16a34a",
    text: "Only the updated model weights travel back. No patient data is sent." },
  { title: "4. Aggregate", color: "#7c3aed",
    text: "The server combines the 4 weight sets into a better global model." },
];
const DUR = [1700, 2200, 1700, 1500];

const NAMES = ["Cleveland", "Hungarian", "Switzerland", "VA"];
const TOTALS = [303, 294, 123, 200];
const POS_RATE = [46, 36, 93, 75];
const HCOL = ["#4f46e5", "#0d9488", "#f97316", "#db2777"];
const HTINT = ["#eef0ff", "#e0f7f4", "#fff1e6", "#fde8f2"];
const HX = [100, 300, 500, 700];
const SERVER = { x: 400, y: 95 };
const HOSP_TOP = 225;

const lerp = (a, b, t) => a + (b - a) * t;

export default function Insights() {
  const [compare, setCompare] = useState({});
  const [base, setBase] = useState({});
  const [algo, setAlgo] = useState("FedCurv");
  const [round, setRound] = useState(1);
  const [phase, setPhase] = useState(0);
  const [prog, setProg] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setCompare(await (await fetch(`${API}/compare`)).json());
        setBase(await (await fetch(`${API}/baselines`)).json());
      } catch {
        setError("Cannot reach the backend. Start it, then refresh.");
      }
    })();
  }, []);

  const data = compare[algo];
  const acc = data ? data.acc : [];
  const R = acc.length;

  const testSizes = base.test_sizes || [61, 59, 25, 40];
  const train = TOTALS.map((t, i) => t - (testSizes[i] || 0));
  const trainSum = train.reduce((a, b) => a + b, 0);
  const share = train.map((n) => Math.round((n / trainSum) * 100));

  // animate progress inside a phase
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setProg((p) => Math.min(1, p + 40 / DUR[phase]));
    }, 40);
    return () => clearInterval(id);
  }, [playing, phase]);

  // move on when a phase finishes
  useEffect(() => {
    if (!playing || prog < 1) return;
    const id = setTimeout(() => advance(true), 250);
    return () => clearTimeout(id);
  }, [playing, prog, phase, round, R]);

  function advance(auto) {
    if (phase < 3) {
      setPhase(phase + 1);
      setProg(auto ? 0 : 1);
    } else if (round < R) {
      setRound(round + 1);
      setPhase(0);
      setProg(auto ? 0 : 1);
    } else {
      setPlaying(false);
    }
  }

  const reset = () => { setPlaying(false); setRound(1); setPhase(0); setProg(0); };
  const play = () => {
    if (round === R && phase === 3 && prog >= 1) reset();
    setProg((p) => (p >= 1 ? 0 : p));
    setPlaying(true);
  };
  const pickAlgo = (a) => { setAlgo(a); reset(); };
  const pickRound = (r) => { setPlaying(false); setRound(r); setPhase(3); setProg(1); };

  if (!data) {
    return (
      <div className="page">
        <h1>How federated learning works</h1>
        <p>{error || "Loading results..."}</p>
      </div>
    );
  }

  const shown = phase === 3 ? acc[round - 1] : round > 1 ? acc[round - 2] : null;
  const bestIdx = acc.indexOf(Math.max(...acc));
  const finalFed = (base.federated || {})[algo];
  const pc = PHASES[phase].color;

  const packet = (i, fromServer, color) => {
    const x = fromServer ? lerp(SERVER.x, HX[i], prog) : lerp(HX[i], SERVER.x, prog);
    const y = fromServer ? lerp(SERVER.y, HOSP_TOP, prog) : lerp(HOSP_TOP, SERVER.y, prog);
    return (
      <g key={`${i}-${fromServer}`}>
        <rect x={x - 14} y={y - 10} width="28" height="20" rx="6" fill={color}
          stroke="#fff" strokeWidth="2" />
        <text x={x} y={y + 4} textAnchor="middle" fontSize="12" fill="#fff" fontWeight="800">W</text>
      </g>
    );
  };

  // accuracy chart
  const CW = 420, CH = 190, PAD = 34;
  const allVals = ALGOS.flatMap((a) => (compare[a] ? compare[a].acc : []));
  const lo = Math.floor(Math.min(...allVals) - 2);
  const hi = Math.ceil(Math.max(...allVals) + 2);
  const px = (i, n) => PAD + (n > 1 ? (i / (n - 1)) * (CW - PAD - 12) : 0);
  const py = (v) => 12 + (1 - (v - lo) / (hi - lo)) * (CH - 12 - 28);
  const line = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${px(i, arr.length)},${py(v)}`).join(" ");

  return (
    <div className="page">
      <h1>How federated learning works</h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        Watch real training rounds from this project. Accuracy values come from the Flower runs.
      </p>

      <div className="row" style={{ marginBottom: 14 }}>
        {ALGOS.map((a) => (
          <button key={a} className={`chip ${algo === a ? "active" : ""}`}
            onClick={() => pickAlgo(a)}>{a}</button>
        ))}
        <span style={{ color: "var(--muted)", fontSize: 14 }}>{ALGO_TEXT[algo]}</span>
      </div>

      <div className="card" style={{ borderTop: `5px solid ${pc}` }}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 6 }}>
          <div>
            <b>Round {round} of {R}</b>
            <span style={{ marginLeft: 12, fontWeight: 700, color: pc }}>
              {PHASES[phase].title}
            </span>
          </div>
          <div className="row">
            {playing
              ? <button className="btn small" onClick={() => setPlaying(false)}>Pause</button>
              : <button className="btn small" onClick={play}>Play</button>}
            <button className="btn ghost small" onClick={() => { setPlaying(false); advance(false); }}>
              Next step
            </button>
            <button className="btn ghost small" onClick={reset}>Reset</button>
          </div>
        </div>
        <p style={{ minHeight: 22 }}>{PHASES[phase].text}</p>

        <svg viewBox="0 0 800 350" style={{ width: "100%", height: "auto" }}>
          <defs>
            <linearGradient id="srv" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#4f46e5" />
              <stop offset="100%" stopColor="#db2777" />
            </linearGradient>
          </defs>

          {HX.map((x, i) => (
            <line key={i} x1={SERVER.x} y1={SERVER.y} x2={x} y2={HOSP_TOP}
              stroke={HCOL[i]} strokeOpacity="0.45" strokeWidth="2.5" strokeDasharray="7 6" />
          ))}

          {/* server */}
          <rect x="285" y="12" width="230" height="83" rx="16" fill="url(#srv)"
            stroke={phase === 3 ? "#facc15" : "none"} strokeWidth="4" />
          <text x="400" y="40" textAnchor="middle" fontSize="16" fontWeight="800" fill="#fff">
            Server: global model
          </text>
          <text x="400" y="60" textAnchor="middle" fontSize="13" fill="#e0e7ff">
            Round {round} / {R}
          </text>
          <text x="400" y="82" textAnchor="middle" fontSize="16" fontWeight="800" fill="#fff">
            {shown === null ? "initial model" : `accuracy ${shown}%`}
          </text>

          {/* hospitals */}
          {HX.map((x, i) => {
            const training = phase === 1;
            return (
              <g key={i}>
                <rect x={x - 88} y={HOSP_TOP} width="176" height="108" rx="14"
                  fill={HTINT[i]} stroke={HCOL[i]} strokeWidth={training ? 4 : 2.5} />
                <rect x={x - 88} y={HOSP_TOP} width="176" height="30" rx="14" fill={HCOL[i]} />
                <rect x={x - 88} y={HOSP_TOP + 16} width="176" height="14" fill={HCOL[i]} />
                <text x={x} y={HOSP_TOP + 21} textAnchor="middle" fontSize="15" fontWeight="800" fill="#fff">
                  {NAMES[i]}
                </text>
                <text x={x} y={HOSP_TOP + 50} textAnchor="middle" fontSize="12.5" fill="#1b2338">
                  {train[i]} training patients
                </text>
                {training ? (
                  <g>
                    <rect x={x - 62} y={HOSP_TOP + 60} width="124" height="9" rx="4.5" fill="#ffffff" />
                    <rect x={x - 62} y={HOSP_TOP + 60} width={124 * prog} height="9" rx="4.5" fill={HCOL[i]} />
                    <text x={x} y={HOSP_TOP + 86} textAnchor="middle" fontSize="12" fill={HCOL[i]} fontWeight="700">
                      training on local data
                    </text>
                  </g>
                ) : (
                  <text x={x} y={HOSP_TOP + 74} textAnchor="middle" fontSize="12" fill="#15803d" fontWeight="700">
                    patient data stays here
                  </text>
                )}
                {(phase === 2 || phase === 3) && (
                  <text x={x} y={HOSP_TOP + 98} textAnchor="middle" fontSize="12" fill="#7c3aed" fontWeight="700">
                    weight in average: {share[i]}%
                  </text>
                )}
              </g>
            );
          })}

          {phase === 0 && HX.map((_, i) => packet(i, true, "#2563eb"))}
          {phase === 2 && HX.map((_, i) => packet(i, false, "#16a34a"))}
        </svg>

        <div className="row" style={{ marginTop: 6 }}>
          <span style={{ fontSize: 14, color: "var(--muted)" }}>Jump to round:</span>
          <input type="range" min="1" max={R} value={round}
            onChange={(e) => pickRound(Number(e.target.value))} style={{ flex: 1, minWidth: 160 }} />
          <b>{round}</b>
        </div>
      </div>

      <div className="grid3" style={{ marginTop: 16 }}>
        <div className="sum-card sum-low">
          <div className="num">0</div>
          <div className="lbl">patient records shared</div>
        </div>
        <div className="sum-card sum-total">
          <div className="num">{R}</div>
          <div className="lbl">communication rounds</div>
        </div>
        <div className="sum-card sum-avg">
          <div className="num">{acc[bestIdx]}%</div>
          <div className="lbl">peak accuracy (round {bestIdx + 1})</div>
        </div>
      </div>

      <div className="grid3" style={{ marginTop: 16, gridTemplateColumns: "1.2fr 1fr" }}>
        <div className="card">
          <h3>Global model accuracy per round</h3>
          <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: "100%", height: "auto" }}>
            {[lo, (lo + hi) / 2, hi].map((v) => (
              <g key={v}>
                <line x1={PAD} x2={CW - 8} y1={py(v)} y2={py(v)} stroke="#e2e8f6" />
                <text x={PAD - 6} y={py(v) + 4} textAnchor="end" fontSize="11" fill="#5d6783">
                  {Math.round(v)}
                </text>
              </g>
            ))}
            {ALGOS.filter((a) => compare[a]).map((a) => (
              <path key={a} d={line(compare[a].acc)} fill="none"
                stroke={ALGO_COLOR[a]} strokeWidth={a === algo ? 3.5 : 1.5}
                opacity={a === algo ? 1 : 0.35} />
            ))}
            {acc.map((v, i) => (
              <text key={i} x={px(i, R)} y={CH - 10} textAnchor="middle" fontSize="10.5" fill="#5d6783">
                {i + 1}
              </text>
            ))}
            {(phase === 3 || round > 1) && (
              <circle cx={px(phase === 3 ? round - 1 : round - 2, R)}
                cy={py(shown ?? acc[0])} r="6" fill="#fff" stroke={ALGO_COLOR[algo]} strokeWidth="3" />
            )}
          </svg>
          <p style={{ fontSize: 13 }}>
            Solid line: {algo}. Faded lines: the other algorithms. X axis: round number.
          </p>
        </div>

        <div className="card">
          <h3>Why it is hard: hospitals differ</h3>
          <p style={{ marginBottom: 10 }}>
            Share of high-risk patients per hospital (approx.). This is non-IID data.
          </p>
          {NAMES.map((n, i) => (
            <div key={n} style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 13, display: "flex", justifyContent: "space-between" }}>
                <span>{n}</span><span>{POS_RATE[i]}%</span>
              </div>
              <div style={{ height: 9, background: "#e9eef6", borderRadius: 5 }}>
                <div style={{ width: `${POS_RATE[i]}%`, height: 9, borderRadius: 5, background: HCOL[i] }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {finalFed && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Final {algo} model on each hospital's own test set</h3>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr><th>Hospital</th><th>Test patients</th><th>Accuracy</th></tr>
              </thead>
              <tbody>
                {NAMES.map((n, i) => (
                  <tr key={n}>
                    <td>
                      <span className="legend-dot" style={{ background: HCOL[i] }} />{n}
                    </td>
                    <td>{testSizes[i]}</td>
                    <td>{finalFed.per_hospital[i]}%</td>
                  </tr>
                ))}
                <tr>
                  <td><b>Overall</b></td>
                  <td><b>{testSizes.reduce((a, b) => a + b, 0)}</b></td>
                  <td><b>{finalFed.overall}%</b></td>
                </tr>
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: 10, fontSize: 13 }}>
            Test sets are small, so one patient moves a hospital's score by several points.
            Accuracy peaked early in training, so more rounds did not always help.
          </p>
        </div>
      )}
    </div>
  );
}