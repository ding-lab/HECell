"""Dataset assembly and cross-validation folds for the baseline.

Joins the per-cell foundation-model features to the Xenium-derived labels and
builds patient-grouped folds so that no patient appears in both train and test.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from config import CELL_DIR, FEATURE_DIR


def load_dataset(groups, classes):
    """Return (X, y, group) aligned row-for-row.

    groups: DataFrame with columns [sample, patient].
    Labels are aligned to the feature row order and filtered to `classes`;
    missing or out-of-vocabulary cells are dropped deterministically.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    feats, labels, grp = [], [], []
    for _, row in groups.iterrows():
        sample, patient = row["sample"], row["patient"]
        fpath = FEATURE_DIR / f"{sample}.pt"
        cpath = CELL_DIR / f"{sample}.parquet"
        if not (fpath.exists() and cpath.exists()):
            print(f"[{sample}] missing features or cell table, skipping", flush=True)
            continue
        blob = torch.load(fpath, map_location="cpu", weights_only=False)
        cells = pd.read_parquet(cpath, columns=["cell_id", "cell_type"])
        cid = np.asarray(blob["cell_id"])
        lab = cells.drop_duplicates("cell_id").set_index("cell_id")["cell_type"].reindex(cid)
        keep = lab.isin(class_to_idx).to_numpy()
        feats.append(blob["feat"][keep].numpy())
        labels.append(lab[keep].map(class_to_idx).to_numpy())
        grp.append(np.full(int(keep.sum()), patient))
    if not feats:
        raise RuntimeError("no samples loaded; check feature/cell-table paths and the groups file")
    X = np.concatenate(feats).astype(np.float32)
    y = np.concatenate(labels).astype(np.int64)
    g = np.concatenate(grp)
    return X, y, g


def make_folds(y, g, n_folds, seed):
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, g))
