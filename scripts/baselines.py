import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federated_learning_app.data import ARTIFACT_DIR, HOSPITALS, INPUT_DIM, load_hospital
from federated_learning_app.task import Net

torch.manual_seed(42)
np.random.seed(42)
EPOCHS, LR, BATCH = 50, 0.01, 32
FED_MODELS = {"FedAvg": "fedavg", "FedProx": "fedprox", "FedCurv": "fedcurv"}


def is_two_class(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, INPUT_DIM))
    return out.ndim > 1 and out.shape[-1] == 2


def probs(model, X):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    if out.ndim > 1 and out.shape[-1] == 2:
        return torch.softmax(out, dim=-1)[:, 1].numpy()
    return torch.sigmoid(out.reshape(-1)).numpy()


def fit(X, y):
    """Train a fresh Net on (X, y) with a plain training loop."""
    model = Net()
    two = is_two_class(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss() if two else nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = order[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = model(Xt[idx])
            if two:
                loss = loss_fn(out, yt[idx].long())
            else:
                loss = loss_fn(out.reshape(-1), yt[idx])
            loss.backward()
            opt.step()
    return model


def acc(p, y):
    return float(((p > 0.5).astype(int) == y.astype(int)).mean() * 100)


def summarize(model, tests):
    """Per-hospital accuracy, overall accuracy and pooled metrics."""
    per, correct, total = [], 0, 0
    all_p, all_y = [], []
    for Xte, yte in tests:
        p = probs(model, Xte)
        per.append(round(acc(p, yte), 2))
        correct += int(((p > 0.5).astype(int) == yte.astype(int)).sum())
        total += len(yte)
        all_p.append(p)
        all_y.append(yte.astype(int))
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    pred = (p > 0.5).astype(int)
    return {
        "per_hospital": per,
        "overall": round(correct / total * 100, 2),
        "precision": round(precision_score(y, pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y, pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y, pred, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y, p) * 100, 2),
    }


splits = [load_hospital(i) for i in range(4)]       # Xtr, Xte, ytr, yte
tests = [(s[1], s[3]) for s in splits]

# 1) Centralized: all hospitals' training data pooled in one place
print("Training centralized model...")
central = fit(np.vstack([s[0] for s in splits]),
              np.concatenate([s[2] for s in splits]))
result = {"hospitals": [h.capitalize() for h in HOSPITALS],
          "test_sizes": [len(t[1]) for t in tests],
          "centralized": summarize(central, tests)}

# 2) Local-only: each hospital trains alone, tested on its own test set
print("Training local-only models...")
local_per, correct, total = [], 0, 0
for i, (Xtr, Xte, ytr, yte) in enumerate(splits):
    m = fit(Xtr, ytr)
    p = probs(m, Xte)
    local_per.append(round(acc(p, yte), 2))
    correct += int(((p > 0.5).astype(int) == yte.astype(int)).sum())
    total += len(yte)
result["local"] = {"per_hospital": local_per,
                   "overall": round(correct / total * 100, 2)}

# 3) Federated models trained earlier with Flower
result["federated"] = {}
for label, key in FED_MODELS.items():
    path = ARTIFACT_DIR / f"{key}_model.pt"
    if not path.exists():
        print("missing:", path.name)
        continue
    m = Net()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    result["federated"][label] = summarize(m, tests)

(ARTIFACT_DIR / "baselines.json").write_text(json.dumps(result, indent=2))

names = result["hospitals"]
print(f"\n{'Model':<14}{'Overall':>9}" + "".join(f"{n[:6]:>9}" for n in names))
rows = [("Centralized", result["centralized"]), ("Local-only", result["local"])]
rows += list(result["federated"].items())
for label, d in rows:
    print(f"{label:<14}{d['overall']:>9}" + "".join(f"{v:>9}" for v in d["per_hospital"]))
print("\nTest sizes:", dict(zip(names, result["test_sizes"])))
print("Saved artifacts/baselines.json")