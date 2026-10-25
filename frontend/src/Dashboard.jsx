import { useEffect, useState } from "react";
import Charts from "./Charts";
import Baselines from "./Baselines";
import StatCards from "./StatCards";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/api";

export default function Dashboard() {
  const [compare, setCompare] = useState({});
  const [baselines, setBaselines] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        setCompare(await (await fetch(`${API}/compare`)).json());
        setBaselines(await (await fetch(`${API}/baselines`)).json());
      } catch {
        setError("Cannot reach the backend. Start it, then refresh.");
      }
    };
    load();
  }, []);

  return (
    <div className="page">
      <h1>Results dashboard</h1>
      <p style={{ color: "var(--muted)", marginTop: 0 }}>
        Your predictions so far, then how the models compare.
      </p>

      <StatCards />

      <h2 className="section-title">Model comparison</h2>
      {error && <p className="error">{error}</p>}
      <div className="card">
        <Baselines data={baselines} />
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <Charts compare={compare} />
      </div>
    </div>
  );
}