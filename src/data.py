"""PTB-XL: build NORM vs MI cohort, split by stratified folds, standardize."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

DATA_DIR = Path("data/ptb-xl")


def load_metadata(data_dir=DATA_DIR):
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    stmts = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    stmts = stmts[stmts["diagnostic"] == 1]

    db["superclass"] = db["scp_codes"].apply(
        lambda codes: {stmts.loc[c, "diagnostic_class"] for c in codes if c in stmts.index}
    )
    return db


def build_cohort(db):
    is_mi = db["superclass"].apply(lambda s: "MI" in s)
    is_norm = db["superclass"].apply(lambda s: s == {"NORM"})
    cohort = db[is_mi | is_norm].copy()
    cohort["label"] = is_mi[cohort.index].astype(int)
    return cohort


def load_signals(cohort, data_dir=DATA_DIR):
    X = np.stack([wfdb.rdsamp(str(data_dir / f))[0] for f in cohort["filename_lr"]])
    return X.transpose(0, 2, 1).astype(np.float32)  # (n, 12 leads, 1000 points)


def split(cohort, X):
    folds = cohort["strat_fold"].to_numpy()
    y = cohort["label"].to_numpy()
    masks = {"train": folds <= 8, "val": folds == 9, "test": folds == 10}
    return {name: (X[m], y[m]) for name, m in masks.items()}


def standardize(splits):
    X_train = splits["train"][0]
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    return {name: ((X - mean) / std, y) for name, (X, y) in splits.items()}


if __name__ == "__main__":
    cohort = build_cohort(load_metadata())
    print(cohort.groupby("strat_fold")["label"].agg(n="count", n_mi="sum"))

    splits = standardize(split(cohort, load_signals(cohort)))

    out = {}
    for name, (X, y) in splits.items():
        out[f"X_{name}"], out[f"y_{name}"] = X, y
    np.savez_compressed("data/processed.npz", **out)