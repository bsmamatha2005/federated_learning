"""Writes artifacts/fl_process.json: facts about the 4 hospitals and the model
updates, used by the dashboard's 'How federated learning works' section."""
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from federated_learning_app.data import ARTIFACT_DIR, HOSPITALS, load_hospital
from federated_learning_app.task import Net

FULL = {
    "cleveland": "Cleveland Clinic, USA",
    "hungarian": "Hungarian Institute of Cardiology",
    "switzerland": "University Hospital, Switzerland",
    "va": "VA Medical Center, Long Beach, USA",
}

hospitals = []
for i, name in enumerate(HOSPITALS):
    Xtr, Xte, ytr, yte = load_hospital(i)
    y = np.concatenate([ytr, yte])
    hospitals.append({
        "id": name, "name": FULL[name], "records": int(len(y)),
        "train": int(len(ytr)), "test": int(len(yte)),
        "positive_pct": round(float(y.mean()) * 100, 1),
    })

cfg = tomllib.load(open(ROOT / "pyproject.toml", "rb"))["tool"]["flwr"]["app"]["config"]
n_params = sum(p.numel() for p in Net().parameters())
info = {
    "hospitals": hospitals,
    "total_records": sum(h["records"] for h in hospitals),
    "model_params": n_params,
    "update_kb": round(n_params * 4 / 1024, 1),           # float32 weights
    "fedcurv_extra_kb": round(2 * n_params * 4 / 1024, 1),  # Fisher + Fisher*weights
    "local_epochs": cfg["local-epochs"],
    "num_rounds": cfg["num-server-rounds"],
}
(ARTIFACT_DIR / "fl_process.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))