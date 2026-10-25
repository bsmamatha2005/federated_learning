import { useEffect, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/api";

const STEPS = [
  ["Send", "The server sends the current model to 4 hospitals."],
  ["Train", "Each hospital trains it on its own patients, locally."],
  ["Return", "Only model weights come back, never patient data."],
  ["Combine", "The server averages the weights into a better model."],
];

export default function Home({ onStart, onInsights }) {
  const [best, setBest] = useState(null);

  useEffect(() => {
    fetch(`${API}/baselines`)
      .then((r) => r.json())
      .then((d) => {
        const f = Object.values(d.federated || {}).map((m) => m.overall);
        if (f.length) setBest(Math.max(...f));
      })
      .catch(() => {});
  }, []);

  const stats = [
    ["4", "Hospitals"],
    ["920", "Patients"],
    ["3", "FL algorithms"],
    [best ? `${best}%` : "...", "Best accuracy"],
  ];

  return (
    <div className="page">
      <div className="hero">
        <h1>Heart attack risk prediction without sharing patient data</h1>
        <p>
          A model trained across 4 hospitals with federated learning (Flower).
          Only weights travel, patient records stay put.
        </p>
        <button className="btn" onClick={onStart}>Try a prediction</button>
        {" "}
        <button className="btn ghost" onClick={onInsights}>See how it works</button>
      </div>

      <div className="grid4">
        {stats.map(([num, lbl]) => (
          <div className="card stat" key={lbl}>
            <div className="num">{num}</div>
            <div className="lbl">{lbl}</div>
          </div>
        ))}
      </div>

      <h2 className="section-title">How one training round works</h2>
      <div className="grid4">
        {STEPS.map(([title, text], i) => (
          <div className="card" key={title}>
            <div className="step-no">{i + 1}</div>
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
        ))}
      </div>

      <p className="footer-note">
        Built with Flower, PyTorch, FastAPI and React. Educational use only,
        not a medical diagnosis.
      </p>
    </div>
  );
}