import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root. HEARTFL_HOME overrides it (needed when Flower runs an installed copy).
ROOT = Path(os.environ.get("HEARTFL_HOME", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

HOSPITALS = ["cleveland", "hungarian", "switzerland", "va"]
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Fixed (mean, std): identical for every hospital and for the API
NUMERIC = {"age": (54, 9), "trestbps": (132, 19), "chol": (200, 110),
           "thalach": (137, 25), "oldpeak": (0.9, 1.1)}
BINARY = ["sex", "fbs", "exang"]
CATS = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2],
        "slope": [1, 2, 3], "thal": [3, 6, 7]}
INPUT_DIM = len(NUMERIC) + len(BINARY) + sum(len(v) for v in CATS.values())  # 21


def preprocess(df: pd.DataFrame) -> np.ndarray:
    """Raw patient columns -> 21 clean numeric features."""
    df = df.copy()
    for c in df.columns:                               # '?' / text -> NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["trestbps", "chol"]:                     # 0 really means "unknown"
        df.loc[df[c] == 0, c] = np.nan

    parts = []
    for c, (mean, std) in NUMERIC.items():             # fill gaps, standardize
        parts.append(((df[c].fillna(mean) - mean) / std).to_numpy()[:, None])
    for c in BINARY:                                   # missing binary -> 0
        parts.append(df[c].fillna(0).to_numpy()[:, None])
    for c, values in CATS.items():                     # one-hot encoding
        for v in values:
            parts.append((df[c] == v).astype(float).to_numpy()[:, None])
    return np.hstack(parts).astype(np.float32)


def load_raw(idx: int) -> pd.DataFrame:
    name = HOSPITALS[idx]
    return pd.read_csv(DATA_DIR / f"processed.{name}.data",
                       header=None, names=COLS, na_values="?")


def load_hospital(idx: int):
    """One hospital's private data -> (X_train, X_test, y_train, y_test)."""
    df = load_raw(idx)
    y = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(np.float32).to_numpy()
    X = preprocess(df.drop(columns=["num", "ca"]))     # ca dropped
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)