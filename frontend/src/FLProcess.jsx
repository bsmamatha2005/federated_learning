import { useEffect, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ReferenceLine,
} from "recharts";

const COLORS = { FedAvg: "#2a7fff", FedProx: "#e8590c", FedCurv: "#2f9e44" };
const card = { border: "1px solid #ccc", borderRadius: 8, padding: 10, background: "#fafafa" };
const SHORT = { cleveland: "Cleveland", hungarian: "Hungarian", switzerland: "Switzerland", va: "VA Long Beach" };

const HOW = {
  FedAvg: "Averages the hospitals' weights, giving more say to hospitals with more patients.",
  FedProx: "Same averaging, but each hospital's training adds a penalty that keeps its model close to the global model, so hospitals with different data do not drift apart.",
  FedCurv: "Same averaging, plus each hospital sends which weights mattered most for its own patients. Other hospitals are penalised for changing those weights.",
};

const PHASES = ["", "Server sends the model", "Hospitals train locally", "Only weights go back", "Server combines them"];

export default function FLProcess({ data, baselines }) {
  const [algo, setAlgo] = useState("FedAvg");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const current = data?.algorithms?.[algo];
  const R = current ? current.rounds.length : data?.num_rounds || 10;
  const maxStep = R * 4;

  useEffect(() => {
    if (!playing) return undefined;
    if (step >= maxStep) { setPlaying(false); return undefined; }
    const t = setTimeout(() => setStep((s) => s + 1), 1400);
    return () => clearTimeout(t);
  }, [playing, step, maxStep]);

  if (!data || !data.hospitals) return <p>Loading federated learning view...</p>;

  const algos = Object.keys(data.algorithms || {});
  const round = step === 0 ? 0 : Math.ceil(step / 4);
  const phase = step === 0 ? 0 : ((step - 1) % 4) + 1;
  const row = round > 0 ? current?.rounds?.[round - 1] : null;
  const prev = round > 1 ? current?.rounds?.[round - 2] : null;
  const shownAcc = phase === 4 ? row : prev;
  const extra = algo === "FedCurv" ? data.fedcurv_extra_kb : 0;

  const fed = baselines?.federated || {};
  const central = baselines?.centralized?.overall;
  const local = baselines?.local?.overall;
  const cell = { border: "1px solid #ccc", padding: 6, verticalAlign: "top" };

  const rs = current?.rounds || [];
  const best = rs.length ? rs.reduce((m, x) => (x.acc > m.acc ? x : m), rs[0]) : null;
  const last = rs.length ? rs[rs.length - 1] : null;

  const caption = [
    "Press Next (or Play) to watch one training round, step by step.",
    `The server sends the current global model (${data.model_params} weights, about ${data.update_kb} KB) to all ${data.hospitals.length} hospitals.`,
    `Each hospital trains the model for ${data.local_epochs} passes (epochs) over its own patients, on its own computer.` +
      (algo === "FedProx" ? ` FedProx also penalises moving far from the global model (mu = ${current?.mu}).` : "") +
      (algo === "FedCurv" ? ` FedCurv also protects weights that other hospitals found important (lambda = ${current?.curv_lambda}).` : ""),
    `Each hospital sends back only its updated weights (about ${data.update_kb} KB)` +
      (algo === "FedCurv" ? ` plus importance values (about ${extra} KB)` : "") +
      ". No patient record ever leaves the hospital.",
    `The server combines the 4 sets of weights into a new global model. ${HOW[algo]}`,
  ][phase];

  const cx = (i) => 97.5 + i * 175;
  const hosp = data.hospitals;
  const down = phase === 1;
  const train = phase === 2;
  const up = phase === 3;
  const merge = phase === 4;

  return (
    <div>
      <style>{`@keyframes flowdash { to { stroke-dashoffset: -28; } }
        .flow { stroke-dasharray: 8 6; animation: flowdash 0.9s linear infinite; }`}</style>

      <h2>How federated learning works in this project</h2>

      <div style={{ ...card, background: "#f1f8ff" }}>
        <strong>What is a round?</strong>
        <div style={{ fontSize: 14, marginTop: 4 }}>
          One round is one full cycle of 4 steps: the server sends the model, every hospital trains it
          on its own patients, the hospitals send back their weights, and the server combines them into a
          better model. We repeat this cycle {data.num_rounds} times, so the model improves round by round.
          Inside each round, a hospital makes {data.local_epochs} passes (epochs) over its own data.
          The patient records are never sent anywhere.
        </div>
      </div>

      <h3>Watch a round happen</h3>
      <div style={{ marginBottom: 8 }}>
        {algos.map((a) => (
          <button key={a} onClick={() => setAlgo(a)}
            style={{
              padding: "6px 12px", marginRight: 6, cursor: "pointer", borderRadius: 6,
              border: `2px solid ${COLORS[a]}`, color: algo === a ? "#fff" : COLORS[a],
              background: algo === a ? COLORS[a] : "#fff",
            }}>
            {a}
          </button>
        ))}
      </div>

      <svg viewBox="0 0 720 400" style={{ width: "100%", border: "1px solid #ddd", borderRadius: 8, background: "#fff" }}>
        <defs>
          <marker id="arrBlue" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
            <path d="M0,0 L12,6 L0,12 z" fill="#2a7fff" />
          </marker>
          <marker id="arrGreen" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
            <path d="M0,0 L12,6 L0,12 z" fill="#2f9e44" />
          </marker>
        </defs>

        <rect x="240" y="15" width="240" height="85" rx="10"
          fill={merge ? "#d3f9d8" : "#e7f0ff"} stroke={merge ? "#2f9e44" : "#2a7fff"} strokeWidth="2" />
        <text x="360" y="42" textAnchor="middle" fontSize="16" fontWeight="bold">Server</text>
        <text x="360" y="64" textAnchor="middle" fontSize="13">
          {round === 0 ? "Global model: random start" : `Global model after round ${phase === 4 ? round : round - 1}`}
        </text>
        <text x="360" y="84" textAnchor="middle" fontSize="13" fill="#555">
          {shownAcc ? `test accuracy ${shownAcc.acc}%` : "no results yet"}
        </text>

        {hosp.map((h, i) => (
          <g key={h.id}>
            {down && (
              <>
                <line className="flow" x1={350 + (i - 1.5) * 12} y1="100" x2={cx(i) - 14} y2="245"
                  stroke="#2a7fff" strokeWidth="3" markerEnd="url(#arrBlue)" />
              </>
            )}
            {up && (
              <>
                <line className="flow" x1={cx(i) + 14} y1="245" x2={370 + (i - 1.5) * 12} y2="100"
                  stroke="#2f9e44" strokeWidth="3" markerEnd="url(#arrGreen)" />
              </>
            )}

            <rect x={cx(i) - 82} y="250" width="165" height="140" rx="10"
              fill={train ? "#fff4e6" : "#fafafa"} stroke={train ? "#e8590c" : "#999"} strokeWidth={train ? 3 : 1.5} />
            <text x={cx(i)} y="272" textAnchor="middle" fontSize="14" fontWeight="bold">{SHORT[h.id] || h.id}</text>
            <text x={cx(i)} y="291" textAnchor="middle" fontSize="12">{h.records} patients</text>
            <rect x={cx(i) - 65} y="301" width="130" height="10" rx="4" fill="#e9ecef" />
            <rect x={cx(i) - 65} y="301" width={(130 * h.positive_pct) / 100} height="10" rx="4" fill="#c92a2a" />
            <text x={cx(i)} y="327" textAnchor="middle" fontSize="12">{h.positive_pct}% have heart disease</text>
            <text x={cx(i)} y="348" textAnchor="middle" fontSize="12" fill="#2b8a3e">Patient data stays here</text>
            <text x={cx(i)} y="371" textAnchor="middle" fontSize="12" fill={train ? "#e8590c" : "#777"}>
              {train ? `training (${data.local_epochs} epochs)...` : up ? "sending weights only" : down ? "receiving model" : "waiting"}
            </text>
          </g>
        ))}
        {(down || up) && (
          <g>
            <rect x="215" y="158" width="290" height="26" rx="13" fill="#fff"
              stroke={down ? "#2a7fff" : "#2f9e44"} />
            <text x="360" y="176" textAnchor="middle" fontSize="13" fontWeight="bold"
              fill={down ? "#2a7fff" : "#2f9e44"}>
              {down ? "global model sent to every hospital"
                : algo === "FedCurv" ? "only weights + importance values" : "only weights, no patient data"}
            </text>
          </g>
        )}
      </svg>

      <div style={{ ...card, marginTop: 8, minHeight: 56 }}>
        <strong>
          {step === 0 ? "Ready" : `Round ${round} of ${R}, step ${phase} of 4: ${PHASES[phase]}`}
        </strong>
        <div style={{ fontSize: 14, marginTop: 4 }}>{caption}</div>
        {row && phase === 4 && (
          <div style={{ fontSize: 14, marginTop: 4 }}>
            After round {round}: test accuracy <strong>{row.acc}%</strong>, test loss {row.eval_loss},
            average training loss {row.train_loss}.
          </div>
        )}
      </div>

      <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={() => { setPlaying(false); setStep((s) => Math.max(0, s - 1)); }}
          style={{ padding: "6px 12px", cursor: "pointer" }}>Back</button>
        <button onClick={() => { setPlaying(false); setStep((s) => Math.min(maxStep, s + 1)); }}
          style={{ padding: "6px 12px", cursor: "pointer" }}>Next</button>
        <button onClick={() => { if (step >= maxStep) setStep(1); setPlaying((p) => !p); }}
          style={{ padding: "6px 12px", cursor: "pointer" }}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setPlaying(false); setStep(0); }}
          style={{ padding: "6px 12px", cursor: "pointer" }}>Reset</button>
        <label style={{ flex: 1, minWidth: 200 }}>
          Jump to round {round || "-"} of {R}
          <input type="range" min={1} max={R} value={Math.max(1, round)}
            onChange={(e) => { setPlaying(false); setStep((Number(e.target.value) - 1) * 4 + 4); }}
            style={{ width: "100%" }} />
        </label>
      </div>

      <h3>Accuracy after each round ({algo})</h3>
      <div style={{ height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={rs.map((x) => ({ round: x.round, accuracy: x.acc }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="round" />
            <YAxis domain={[60, 100]} />
            <Tooltip />
            <Line type="monotone" dataKey="accuracy" stroke={COLORS[algo]} strokeWidth={2} dot />
            {round > 0 && <ReferenceLine x={round} stroke="#c92a2a" strokeDasharray="4 4" />}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {best && (
        <p style={{ fontSize: 13, color: "#555" }}>
          Best accuracy {best.acc}% at round {best.round}; final {last.acc}% at round {last.round}.
          On this small dataset the model learns most in the first few rounds, and more rounds
          do not always help.
        </p>
      )}

      <h3>Normal training vs federated training</h3>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={cell}></th>
            <th style={cell}>Normal (centralized)</th>
            <th style={cell}>Federated</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={cell}>Where patient data goes</td>
            <td style={cell}>Copied to one central server</td>
            <td style={cell}>Stays in each hospital</td>
          </tr>
          <tr>
            <td style={cell}>What is shared</td>
            <td style={cell}>All patient records</td>
            <td style={cell}>Only model weights</td>
          </tr>
          <tr>
            <td style={cell}>Accuracy in this project</td>
            <td style={cell}>{central !== undefined ? `${central}%` : "-"}</td>
            <td style={cell}>
              {algos.length === 0 ? "-" : algos.map((a) => `${a} ${fed[a]?.overall ?? "-"}%`).join(", ")}
            </td>
          </tr>
          <tr>
            <td style={cell}>Training</td>
            <td style={cell}>One machine, one run</td>
            <td style={cell}>{data.num_rounds} rounds of coordination between hospitals and server</td>
          </tr>
        </tbody>
      </table>
      <p style={{ fontSize: 13, color: "#555" }}>
        On this small dataset, federated accuracy is close to centralized accuracy
        {local !== undefined ? ` (training only on each hospital's own data gives ${local}%)` : ""}.
        The differences are a few patients, so the main benefit shown here is privacy, not higher accuracy.
      </p>

      <h3>The three algorithms compared</h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
        {algos.map((a) => {
          const r2 = data.algorithms[a].rounds;
          const b = r2.reduce((m, x) => (x.acc > m.acc ? x : m), r2[0]);
          const l = r2[r2.length - 1];
          return (
            <div key={a} style={{ ...card, borderTop: `4px solid ${COLORS[a]}` }}>
              <strong>{a}</strong>
              <div style={{ fontSize: 14, margin: "4px 0" }}>{HOW[a]}</div>
              <div style={{ fontSize: 13 }}>
                Best: {b.acc}% (round {b.round})<br />
                Final: {l.acc}% (round {l.round})
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}