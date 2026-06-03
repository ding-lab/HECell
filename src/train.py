#!/usr/bin/env python
"""Train the baseline classifier under patient-stratified cross-validation.

For each fold, standardizes the features on the training split, trains the MLP
with a class-weighted loss, and records out-of-fold probabilities. Writes the
out-of-fold predictions and one checkpoint per fold (with the standardization
statistics needed at inference).
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODEL_DIR, OUT, load_config
from data import load_dataset, make_folds
from model import MLP


def train_fold(Xtr, ytr, Xva, classes, tr, device):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    counts = np.bincount(ytr, minlength=len(classes))
    weight = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)),
                          dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=tr["batch_size"], shuffle=True)
    model = MLP(Xtr.shape[1], len(classes), tr["hidden"], tr["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=tr["label_smoothing"])
    for _ in range(tr["epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.from_numpy(Xva).to(device)), 1).cpu().numpy()
    ckpt = {"state_dict": model.state_dict(), "mu": mu, "sd": sd,
            "classes": classes, "hidden": tr["hidden"], "dropout": tr["dropout"]}
    return proba, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--groups", required=True, help="CSV with columns: sample, patient")
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes, tr, cv = cfg["classes"], cfg["train"], cfg["cv"]
    torch.manual_seed(cv["seed"])
    np.random.seed(cv["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    groups = pd.read_csv(args.groups)
    X, y, g = load_dataset(groups, classes)
    print(f"{len(y):,} cells, dim={X.shape[1]}, "
          f"{dict(zip(classes, np.bincount(y, minlength=len(classes)).tolist()))}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    proba = np.zeros((len(y), len(classes)), dtype=np.float32)
    for k, (tr_idx, va_idx) in enumerate(make_folds(y, g, cv["n_folds"], cv["seed"])):
        proba[va_idx], ckpt = train_fold(X[tr_idx], y[tr_idx], X[va_idx], classes, tr, device)
        torch.save(ckpt, MODEL_DIR / f"fold{k}.pt")
        print(f"fold {k} done ({len(va_idx):,} held-out cells)", flush=True)

    oof = pd.DataFrame({"y_true": y})
    for i, c in enumerate(classes):
        oof[f"p_{c}"] = proba[:, i]
    OUT.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUT / "oof_predictions.parquet")
    print(f"wrote {OUT / 'oof_predictions.parquet'} and {cv['n_folds']} fold checkpoints", flush=True)


if __name__ == "__main__":
    main()
