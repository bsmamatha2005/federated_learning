import { useEffect, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/api";

function Donut({ high, low }) {
  const total = high + low;
  const R = 52, C = 2 * Math.PI * R;
  const highLen = total ? (high / total) * C : 0;
  return (
    <svg viewBox="0 0 140 140" width="140" height="140">
      <circle cx="70" cy="70" r={R} fill="none" stroke="#16a34a" strokeWidth="18" />
      <circle cx="70" cy="70" r={R} fill="none" stroke="#e11d48" strokeWidth="18"
        strokeDasharray={`${highLen} ${C - highLen}`} transform="rotate(-90 70 70)" />
      <text x="70" y="68" textAnchor="middle" fontSize="24" fontWeight="800" fill="#1b2338">
        {total}
      </text>
      <text x="70" y="86" textAnchor="middle" fontSize="11" fill="#5d6783">predictions</text>
    </svg>
  );
}

export default function StatCards({ refreshKey = 0 }) {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/history`)
      .then((r) => r.json())
      .then((d) => { setRows(Array.isArray(d) ? d : []); setError(""); })
      .catch(() => setError("Cannot reach the backend."));
  }, [refreshKey]);

  const total = rows.length;
  const high = rows.filter((r) => r.binary === 1).length;
  const low = total - high;
  const avg = total ? (rows.reduce((s, r) => s + r.probability, 0) / total) * 100 : 0;

  const cards = [
    ["sum-total", total, "Total predictions"],
    ["sum-high", high, "High risk"],
    ["sum-low", low, "Low risk"],
    ["sum-avg", `${avg.toFixed(1)}%`, "Average probability"],
  ];

  return (
    <div>
      <div className="grid4">
        {cards.map(([cls, num, lbl]) => (
          <div className={`sum-card ${cls}`} key={lbl}>
            <div className="num">{num}</div>
            <div className="lbl">{lbl}</div>
          </div>
        ))}
      </div>
      {error && <p className="error">{error}</p>}
      <div className="card donut-wrap" style={{ marginTop: 16 }}>
        <Donut high={high} low={low} />
        <div>
          <h3>High vs low risk</h3>
          <p>
            <span className="legend-dot" style={{ background: "#e11d48" }} />
            High risk: <b>{high}</b>
          </p>
          <p style={{ marginTop: 6 }}>
            <span className="legend-dot" style={{ background: "#16a34a" }} />
            Low risk: <b>{low}</b>
          </p>
        </div>
      </div>
    </div>
  );
}