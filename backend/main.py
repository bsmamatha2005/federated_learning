import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from federated_learning_app.data import ARTIFACT_DIR, preprocess
from federated_learning_app.task import Net

app = FastAPI(title="Federated Heart Risk API")

origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"])

# Which trained model to serve: fedavg, fedprox or fedcurv
MODEL_NAME = os.environ.get("MODEL_NAME", "fedcurv")
model = Net()
model.load_state_dict(torch.load(ARTIFACT_DIR / f"{MODEL_NAME}_model.pt",
                                 map_location="cpu"))
model.eval()

DB = ROOT / "backend" / "history.db"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "thal"]
LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcurv": "FedCurv"}


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT,
        inputs TEXT, probability REAL, risk TEXT, binary INTEGER)""")
    return con


class Patient(BaseModel):
    age: float
    sex: int            # 1 male, 0 female
    cp: int             # chest pain type 1-4
    trestbps: float     # resting blood pressure
    chol: float         # cholesterol
    fbs: int            # fasting sugar > 120: 1/0
    restecg: int        # 0-2
    thalach: float      # max heart rate
    exang: int          # exercise angina 1/0
    oldpeak: float      # ST depression
    slope: int          # 1-3
    thal: int           # 3, 6 or 7


def risk_probability(x):
    """Works whether Net outputs 1 logit or 2 class scores."""
    with torch.no_grad():
        out = model(x)
    if out.ndim > 1 and out.shape[-1] == 2:
        return torch.softmax(out, dim=-1)[0, 1].item()
    return torch.sigmoid(out.reshape(-1)[0]).item()


@app.get("/")
def health():
    return {"status": "ok", "service": "federated-heart-api",
            "model": LABELS.get(MODEL_NAME, MODEL_NAME)}


@app.post("/api/predict")
def predict(p: Patient):
    row = p.model_dump()
    x = torch.tensor(preprocess(pd.DataFrame([row])[FEATURES]))
    prob = risk_probability(x)
    binary = int(prob > 0.5)
    risk = "High" if binary else "Low"
    con = db()
    con.execute("INSERT INTO predictions(created, inputs, probability, risk, binary) "
                "VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), json.dumps(row),
                 prob, risk, binary))
    con.commit()
    con.close()
    return {"probability": round(prob, 4), "risk": risk, "binary": binary,
            "model": LABELS.get(MODEL_NAME, MODEL_NAME)}


@app.get("/api/history")
def history():
    con = db()
    rows = con.execute("SELECT id, created, inputs, probability, risk, binary "
                       "FROM predictions ORDER BY id DESC LIMIT 100").fetchall()
    con.close()
    return [{"id": r[0], "created": r[1], "inputs": json.loads(r[2]),
             "probability": r[3], "risk": r[4], "binary": r[5]} for r in rows]


@app.get("/api/compare")
def compare():
    """Per-round test accuracy (%) and loss for FedAvg, FedProx, FedCurv."""
    out = {}
    for key, label in LABELS.items():
        f = ARTIFACT_DIR / f"{key}_history.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text()).get("evaluate_metrics_clientapp", {})
        rounds = sorted(int(r) for r in ev)
        out[label] = {
            "rounds": rounds,
            "acc": [round(ev[str(r)]["eval_acc"] * 100, 2) for r in rounds],
            "loss": [round(ev[str(r)]["eval_loss"], 4) for r in rounds],
        }
    return out

@app.get("/api/baselines")
def baselines():
    """Centralized vs local-only vs federated results (from scripts/baselines.py)."""
    f = ARTIFACT_DIR / "baselines.json"
    return json.loads(f.read_text()) if f.exists() else {}



@app.get("/api/fl-process")
def fl_process():
    """Hospital facts + per-round results, for the 'How federated learning works' section."""
    f = ARTIFACT_DIR / "fl_process.json"
    info = json.loads(f.read_text()) if f.exists() else {}
    algorithms = {}
    for key, label in LABELS.items():
        hf = ARTIFACT_DIR / f"{key}_history.json"
        if not hf.exists():
            continue
        h = json.loads(hf.read_text())
        ev = h.get("evaluate_metrics_clientapp", {})
        tr = h.get("train_metrics_clientapp", {})
        algorithms[label] = {
            "mu": h.get("mu", 0),
            "curv_lambda": h.get("curv_lambda", 0),
            "rounds": [
                {"round": r,
                 "acc": round(ev[str(r)]["eval_acc"] * 100, 2),
                 "eval_loss": round(ev[str(r)]["eval_loss"], 4),
                 "train_loss": round(tr.get(str(r), {}).get("train_loss", 0), 4)}
                for r in sorted(int(x) for x in ev)
            ],
        }
    info["algorithms"] = algorithms
    return info