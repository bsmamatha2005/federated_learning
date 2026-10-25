import { useEffect, useState } from "react";
import StatCards from "./StatCards";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/api";

export default function History() {
  const [rows, setRows] = useState([]);
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);

  const load = async () => {
    try {
      const d = await (await fetch(`${API}/history`)).json();
      setRows(Array.isArray(d) ? d : []);
      setError("");
    } catch {
      setError("Cannot reach the backend. Start it, then refresh.");
    }
    setVersion((v) => v + 1);
  };
  useEffect(() => { load(); }, []);

  const remove = async (id) => {
    if (!window.confirm("Delete this record?")) return;
    await fetch(`${API}/history/${id}`, { method: "DELETE" });
    load();
  };

  const clearAll = async () => {
    if (!window.confirm("Delete ALL records? This cannot be undone.")) return;
    await fetch(`${API}/history`, { method: "DELETE" });
    load();
  };

  const shown = rows.filter((r) => {
    if (filter === "high" && !r.binary) return false;
    if (filter === "low" && r.binary) return false;
    const q = query.trim().toLowerCase();
    return !q || r.created.toLowerCase().includes(q) || String(r.inputs.age).includes(q);
  });

  return (
    <div className="page">
      <h1>Prediction history</h1>

      <StatCards refreshKey={version} />

      <div className="toolbar">
        <input placeholder="Search by age or date" value={query}
          onChange={(e) => setQuery(e.target.value)} />
        {["all", "high", "low"].map((f) => (
          <button key={f} className={`chip ${filter === f ? "active" : ""}`}
            onClick={() => setFilter(f)}>
            {f === "all" ? "All" : f === "high" ? "High risk" : "Low risk"}
          </button>
        ))}
        <a className="btn ghost small" href={`${API}/history/export`}
          style={{ textDecoration: "none" }}>Export CSV</a>
        <button className="btn danger small" onClick={clearAll}>Clear all</button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="card table-wrap" style={{ padding: 0 }}>
        <table className="data">
          <thead>
            <tr>
              <th>Time</th><th>Age</th><th>Sex</th><th>BP</th><th>Chol</th>
              <th>Probability</th><th>Risk</th><th></th>
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 && (
              <tr><td colSpan="8" className="empty">No records to show.</td></tr>
            )}
            {shown.map((r) => (
              <tr key={r.id}>
                <td>{r.created.replace("T", " ")}</td>
                <td>{r.inputs.age}</td>
                <td>{r.inputs.sex === 1 ? "M" : "F"}</td>
                <td>{r.inputs.trestbps}</td>
                <td>{r.inputs.chol}</td>
                <td>{(r.probability * 100).toFixed(1)}%</td>
                <td>
                  <span className={`badge ${r.binary ? "high" : "low"}`}>{r.risk}</span>
                </td>
                <td>
                  <button className="btn danger small" onClick={() => remove(r.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}