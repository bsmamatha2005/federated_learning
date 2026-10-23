import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from federated_learning_app.data import HOSPITALS, INPUT_DIM, load_hospital, load_raw, preprocess

pd.set_option("display.width", 200)

print("=== RAW (Cleveland, first 3 rows) ===")
raw = load_raw(0)
print(raw.head(3).to_string())

print("\n=== MISSING VALUES PER HOSPITAL ===")
rows = []
for i, name in enumerate(HOSPITALS):
    r = load_raw(i)
    rows.append({"hospital": name, "records": len(r),
                 "chol==0": int((r["chol"] == 0).sum()),
                 "chol missing(?)": int(r["chol"].isna().sum()),
                 "ca missing": int(r["ca"].isna().sum())})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== PREPROCESSED (first Cleveland row) ===")
X = preprocess(raw.drop(columns=["num", "ca"]))
print("shape:", X.shape, "(expected (303, 21))")
print(np.round(X[0], 2))

print("\n=== PER-HOSPITAL SPLIT ===")
print(f"{'hospital':<13}{'train':>6}{'test':>6}{'positive %':>12}{'NaN left':>10}")
total = 0
for i, name in enumerate(HOSPITALS):
    Xtr, Xte, ytr, yte = load_hospital(i)
    total += len(Xtr) + len(Xte)
    assert Xtr.shape[1] == INPUT_DIM
    pos = np.concatenate([ytr, yte]).mean() * 100
    nan = int(np.isnan(Xtr).sum() + np.isnan(Xte).sum())
    print(f"{name:<13}{len(Xtr):>6}{len(Xte):>6}{pos:>11.1f}%{nan:>10}")
print("\nTotal records:", total)