import { useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/api";

const SAMPLES = {
  high: { age: 63, sex: 1, cp: 4, trestbps: 150, chol: 280, fbs: 1, restecg: 2,
          thalach: 110, exang: 1, oldpeak: 3, slope: 2, thal: 7 },
  low: { age: 35, sex: 0, cp: 3, trestbps: 120, chol: 190, fbs: 0, restecg: 0,
         thalach: 170, exang: 0, oldpeak: 0, slope: 1, thal: 3 },
};

const NUMBERS = [
  ["age", "Age (years)", 20, 90, 1],
  ["trestbps", "Resting BP (mm Hg)", 80, 220, 1],
  ["chol", "Cholesterol (mg/dl)", 100, 600, 1],
  ["thalach", "Max heart rate", 60, 220, 1],
  ["oldpeak", "ST depression", 0, 7, 0.1],
];

const CHOICES = [
  ["sex", "Sex", [[1, "Male"], [0, "Female"]]],
  ["cp", "Chest pain type", [[1, "Typical angina"], [2, "Atypical angina"],
                             [3, "Non-anginal"], [4, "Asymptomatic"]]],
  ["fbs", "Fasting sugar > 120", [[0, "No"], [1, "Yes"]]],
  ["restecg", "Resting ECG", [[0, "Normal"], [1, "ST-T abnormality"], [2, "LV hypertrophy"]]],
  ["exang", "Exercise angina", [[0, "No"], [1, "Yes"]]],
  ["slope", "ST slope", [[1, "Upsloping"], [2, "Flat"], [3, "Downsloping"]]],
  ["thal", "Thalassemia", [[3, "Normal"], [6, "Fixed defect"], [7, "Reversible defect"]]],
];

export default function Predict() {
  const [form, setForm] = useState(SAMPLES.high);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const set = (k, v) => setForm({ ...form, [k]: v });

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const body = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, Number(v)])
      );
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("bad input");
      setResult(await res.json());
    } catch {
      setError("Prediction failed. Check the values and that the backend is running.");
    }
    setBusy(false);
  };

  const download = () => {
    const lines = [
      "HEART ATTACK RISK REPORT",
      `Date: ${new Date().toLocaleString()}`,
      `Model: ${result.model} (federated learning, 4 hospitals)`,
      "",
      ...Object.entries(form).map(([k, v]) => `${k}: ${v}`),
      "",
      `Risk: ${result.risk}`,
      `Probability: ${(result.probability * 100).toFixed(1)}%`,
      "",
      "Educational use only, not a medical diagnosis.",
    ];
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    a.download = "risk_report.txt";
    a.click();
  };

  const high = result && result.binary === 1;
  const pct = result ? (result.probability * 100).toFixed(1) : 0;

  return (
    <div className="page">
      <h1 style={{ marginTop: 0 }}>Predict risk</h1>

      <div className="card">
        <form onSubmit={submit}>
          <div className="row" style={{ marginBottom: 12 }}>
            <span style={{ color: "var(--muted)", fontSize: 14 }}>Quick fill:</span>
            <button type="button" className="chip" onClick={() => setForm(SAMPLES.high)}>
              High-risk sample
            </button>
            <button type="button" className="chip" onClick={() => setForm(SAMPLES.low)}>
              Healthy sample
            </button>
          </div>

          <div className="sub-title">Measurements</div>
          <div className="form-grid">
            {NUMBERS.map(([k, label, min, max, step]) => (
              <div className="field" key={k}>
                <label>{label}</label>
                <input type="number" required min={min} max={max} step={step}
                  value={form[k]} onChange={(e) => set(k, e.target.value)} />
              </div>
            ))}
          </div>

          <div className="sub-title">Clinical details</div>
          <div className="form-grid">
            {CHOICES.map(([k, label, options]) => (
              <div className="field" key={k}>
                <label>{label}</label>
                <select value={form[k]} onChange={(e) => set(k, e.target.value)}>
                  {options.map(([v, text]) => (
                    <option key={v} value={v}>{text}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="row" style={{ marginTop: 20 }}>
            <button className="btn" type="submit" disabled={busy}>
              {busy ? "Predicting..." : "Predict"}
            </button>
          </div>
          {error && <p className="error">{error}</p>}
        </form>
      </div>

      {result && (
        <div className={`card result ${high ? "high" : ""}`}>
          <div style={{ color: "var(--muted)", fontSize: 14 }}>Result</div>
          <div className={`big ${high ? "high" : "low"}`}>
            {result.risk} risk, {pct}%
          </div>
          <div className="gauge">
            <div className="marker" style={{ left: `calc(${pct}% - 2px)` }} />
          </div>
          <div className="gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>
          <p style={{ marginTop: 12 }}>
            Model: <b>{result.model}</b> (trained across 4 hospitals).
            Educational use only, not a diagnosis.
          </p>
          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn ghost" onClick={download}>Download report</button>
          </div>
        </div>
      )}
    </div>
  );
}