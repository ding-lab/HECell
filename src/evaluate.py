#!/usr/bin/env python
"""Score the out-of-fold predictions and render the result plots.

Reads the out-of-fold predictions written by train.py, computes macro-F1,
per-class F1, macro-AUC and the confusion matrix, and saves a metrics file plus
a per-class-F1 bar chart and a row-normalized confusion matrix.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from config import FIG_DIR, OUT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--oof", default=None, help="out-of-fold predictions parquet")
    args = ap.parse_args()

    classes = load_config(args.config)["classes"]
    df = pd.read_parquet(args.oof or (OUT / "oof_predictions.parquet"))
    y = df["y_true"].to_numpy()
    proba = df[[f"p_{c}" for c in classes]].to_numpy()
    pred = proba.argmax(1)

    labels = range(len(classes))
    per_class = f1_score(y, pred, average=None, labels=labels, zero_division=0)
    try:
        macro_auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        macro_auc = float("nan")   # a class absent from this run's labels
    metrics = {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "macro_auc": macro_auc,
        "per_class_f1": {c: float(v) for c, v in zip(classes, per_class)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    bars = ax.bar(range(len(classes)), per_class, 0.6, color="#2c7fb8")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .006,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_ylabel("F1 (per class)"); ax.set_ylim(0, 1.0); ax.set_title("Per-class F1 (baseline)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "per_class_f1.png", dpi=150); plt.close()

    cm = confusion_matrix(y, pred, labels=range(len(classes))).astype(float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (Xenium)")
    ax.set_title("Confusion matrix (baseline, row-normalized)")
    plt.colorbar(im, fraction=0.046); plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150); plt.close()
    print(f"wrote metrics and figures under {OUT}", flush=True)


if __name__ == "__main__":
    main()
