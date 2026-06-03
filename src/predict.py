#!/usr/bin/env python
"""Apply a trained fold checkpoint to one sample's features.

Loads a checkpoint (with its standardization statistics), scores every cell in
the sample, and writes the predicted type and confidence per cell.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from config import FEATURE_DIR, PRED_DIR
from model import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model", required=True, help="path to a fold checkpoint from train.py")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    classes, mu, sd = ckpt["classes"], ckpt["mu"], ckpt["sd"]
    model = MLP(mu.shape[1], len(classes), ckpt["hidden"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    blob = torch.load(FEATURE_DIR / f"{args.sample}.pt", map_location="cpu", weights_only=False)
    cid = np.asarray(blob["cell_id"])
    feat = (blob["feat"].numpy() - mu) / sd
    with torch.no_grad():
        proba = torch.softmax(model(torch.from_numpy(feat).float()), 1).numpy()

    idx = proba.argmax(1)
    out = pd.DataFrame({
        "cell_id": cid,
        "pred_type": [classes[i] for i in idx],
        "confidence": proba.max(1),
    })
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_DIR / f"{args.sample}.parquet")
    print(f"[{args.sample}] {len(out)} cells -> predictions/{args.sample}.parquet", flush=True)


if __name__ == "__main__":
    main()
