import { useState } from "react";
import Home from "./Home";
import Predict from "./Predict";
import History from "./History";
import Insights from "./Insights";
import Dashboard from "./Dashboard";

const PAGES = [
  ["home", "Home"],
  ["predict", "Predict"],
  ["history", "History"],
  ["insights", "How it works"],
  ["dashboard", "Results"],
];

export default function App() {
  const [page, setPage] = useState("home");

  const go = (p) => {
    setPage(p);
    window.scrollTo(0, 0);
  };

  return (
    <div>
      <nav className="nav">
        <div className="brand" onClick={() => go("home")}>FedHeart</div>
        <div className="nav-links">
          {PAGES.map(([id, label]) => (
            <button key={id}
              className={`nav-link ${page === id ? "active" : ""}`}
              onClick={() => go(id)}>
              {label}
            </button>
          ))}
        </div>
      </nav>

      {page === "home" && (
        <Home onStart={() => go("predict")} onInsights={() => go("insights")} />
      )}
      {page === "predict" && <Predict />}
      {page === "history" && <History />}
      {page === "insights" && <Insights />}
      {page === "dashboard" && <Dashboard />}
    </div>
  );
}