import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parents[1] / "artifacts"
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcurv": "FedCurv"}

runs = {}
for key, label in NAMES.items():
    path = ART / f"{key}_history.json"
    if not path.exists():
        print("missing:", path.name)
        continue
    ev = json.load(open(path)).get("evaluate_metrics_clientapp", {})
    rounds = sorted(int(r) for r in ev)
    runs[label] = {
        "rounds": rounds,
        "acc": [ev[str(r)]["eval_acc"] * 100 for r in rounds],
        "loss": [ev[str(r)]["eval_loss"] for r in rounds],
    }

rows = []
print(f"\n{'Algorithm':<10}{'Final acc %':>12}{'Best acc %':>12}{'Best round':>11}"
      f"{'Avg last 3 %':>14}{'Final loss':>12}")
for label, d in runs.items():
    best = max(d["acc"])
    row = [label, round(d["acc"][-1], 2), round(best, 2),
           d["rounds"][d["acc"].index(best)],
           round(sum(d["acc"][-3:]) / 3, 2), round(d["loss"][-1], 4)]
    rows.append(row)
    print(f"{row[0]:<10}{row[1]:>12}{row[2]:>12}{row[3]:>11}{row[4]:>14}{row[5]:>12}")

with open(ART / "comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algorithm", "final_acc", "best_acc", "best_round", "avg_last3_acc", "final_loss"])
    w.writerows(rows)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for label, d in runs.items():
    ax[0].plot(d["rounds"], d["acc"], marker="o", label=label)
    ax[1].plot(d["rounds"], d["loss"], marker="o", label=label)
ax[0].set(title="Test accuracy per round", xlabel="Round", ylabel="Accuracy (%)")
ax[1].set(title="Test loss per round", xlabel="Round", ylabel="Loss")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()
plt.savefig(ART / "comparison.png", dpi=150)
print("\nSaved artifacts/comparison.png and comparison.csv")