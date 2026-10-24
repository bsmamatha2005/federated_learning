import { useState } from "react";

const COLORS = { FedAvg: "#2a7fff", FedProx: "#e8590c", FedCurv: "#2f9e44" };
const card = { border: "1px solid #ccc", borderRadius: 8, padding: 10, background: "#fafafa" };

const HOW = {
  FedAvg: "Averages the hospitals' weights, giving more say to hospitals with more patients.",
  FedProx: "Same averaging, but each hospital's training adds a penalty that keeps its model close to the global model, so hospitals with different data do not drift apart.",
  FedCurv: "Same averaging, plus each hospital sends which weights mattered most for its own patients. Other hospitals are penalised for changing those weights.",
};

export default function FLProcess({ data, baselines }) {
  const [algo, setAlgo] = useState("FedAvg");
  const [round, setRound] = useState(1);

  if (!data || !data.hospitals) return <p>Loading federated learning view...</p>;

  const algos = Object.keys(data.algorithms || {});
  const current = data.algorithms?.[algo];
  const maxRound = current ? current.rounds.length : data.num_rounds;
  const r = Math.min(round, maxRound);
  const row = current?.rounds?.[r - 1];
  const extra = algo === "FedCurv" ? data.fedcurv_extra_kb : 0;
  const perRoundKb = (data.update_kb * 2 + extra) * data.hospitals.length;

  const fed = baselines?.federated || {};
  const central = baselines?.centralized?.overall;
  const local = baselines?.local?.overall;

  const cell = { border: "1px solid #ccc", padding: 6, verticalAlign: "top" };

  return (
    <div>
      <h2>How federated learning works in this project</h2>

      <h3>1. Four hospitals, and their data never leaves them</h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        {data.hospitals.map((h) => (
          <div key={h.id} style={card}>
            <strong>{h.name}</strong>
            <div>{h.records} patients ({h.train} train, {h.test} test)</div>
            <div style={{ margin: "6px 0", background: "#e9ecef", borderRadius: 4, height: 12 }}>
              <div style={{ width: `${h.positive_pct}%`, height: 12, borderRadius: 4, background: "#c92a2a" }} />
            </div>
            <div style={{ fontSize: 13 }}>{h.positive_pct}% have heart disease</div>
            <div style={{ fontSize: 13, color: "#2b8a3e", marginTop: 4 }}>
              Patient data stays in this hospital
            </div>
          </div>
        ))}
      </div>
      <p style={{ fontSize: 13, color: "#555" }}>
        The disease rate runs from {Math.min(...data.hospitals.map((h) => h.positive_pct))}% to{" "}
        {Math.max(...data.hospitals.map((h) => h.positive_pct))}%. Hospitals with such different
        data are called non-IID, and that is what makes federated learning hard.
      </p>

      <h3>2. What happens in one training round</h3>
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
      <label>
        Round {r} of {maxRound}
        <input type="range" min={1} max={maxRound} value={r}
          onChange={(e) => setRound(Number(e.target.value))}
          style={{ width: "100%" }} />
      </label>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 8 }}>
        <div style={card}>
          <strong>Step 1: server sends the model</strong>
          <div style={{ fontSize: 14 }}>
            The current global model ({data.model_params} weights, about {data.update_kb} KB)
            goes to all {data.hospitals.length} hospitals.
          </div>
        </div>
        <div style={card}>
          <strong>Step 2: each hospital trains locally</strong>
          <div style={{ fontSize: 14 }}>
            {data.local_epochs} passes over its own patients, on its own computer.
            {algo === "FedProx" && ` Penalty strength mu = ${current?.mu}.`}
            {algo === "FedCurv" && ` Penalty strength lambda = ${current?.curv_lambda}.`}
          </div>
        </div>
        <div style={card}>
          <strong>Step 3: only weights come back</strong>
          <div style={{ fontSize: 14 }}>
            Each hospital returns its updated weights (about {data.update_kb} KB).
            {algo === "FedCurv" && ` FedCurv also returns importance values (about ${data.fedcurv_extra_kb} KB more).`}
            {" "}No patient record is sent. About {perRoundKb.toFixed(0)} KB moves in total this round.
          </div>
        </div>
        <div style={card}>
          <strong>Step 4: server combines them</strong>
          <div style={{ fontSize: 14 }}>{HOW[algo]}</div>
        </div>
      </div>

      {row && (
        <p style={{ marginTop: 8 }}>
          <strong>After round {r}:</strong> test accuracy {row.acc}%, test loss {row.eval_loss},
          average training loss {row.train_loss}
        </p>
      )}

      <h3>3. Normal training vs federated training</h3>
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
              {algos.length === 0 ? "-" : algos.map((a) =>
                `${a} ${fed[a]?.overall ?? "-"}%`).join(", ")}
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

      <h3>4. The three algorithms compared</h3>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
        {algos.map((a) => {
          const rs = data.algorithms[a].rounds;
          const best = rs.reduce((m, x) => (x.acc > m.acc ? x : m), rs[0]);
          const last = rs[rs.length - 1];
          return (
            <div key={a} style={{ ...card, borderTop: `4px solid ${COLORS[a]}` }}>
              <strong>{a}</strong>
              <div style={{ fontSize: 14, margin: "4px 0" }}>{HOW[a]}</div>
              <div style={{ fontSize: 13 }}>
                Best: {best.acc}% (round {best.round})<br />
                Final: {last.acc}% (round {last.round})
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}