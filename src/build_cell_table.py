#!/usr/bin/env python
"""Build the per-cell table used by the baseline.

For each sample, reads a Xenium cell table (cell id, centroid in microns, and a
cell-type label), maps the labels onto the target classes, converts the
centroids to pixel coordinates on the aligned H&E, and writes the per-cell table
that extract_features.py consumes.

Input  : $HECELL_ROOT/data/xenium/<sample>.parquet
           columns: cell_id, x_centroid, y_centroid, cell_type   (centroids in microns)
Output : $HECELL_ROOT/data/cells/<sample>.parquet
           columns: cell_id, x_px, y_px, cell_type
"""
import argparse

import pandas as pd

from config import CELL_DIR, XENIUM_DIR, load_config

# Map the raw annotation labels onto the target classes. Edit this to match the
# label vocabulary in your own Xenium cell table.
LABEL_MAP = {
    "Tumor": "Tumor",
    "Hepatocyte": "Hepatocyte",
    "Stromal": "Stromal",
    "Fibroblast": "Stromal",
    "Endothelial": "Stromal",
    "Lymphocyte": "Lymphocyte",
    "T cell": "Lymphocyte",
    "B cell": "Lymphocyte",
    "Necrosis": "Necrosis",
}


def build(sample, classes, microns_per_px):
    df = pd.read_parquet(XENIUM_DIR / f"{sample}.parquet")
    df["cell_type"] = df["cell_type"].map(LABEL_MAP)
    df = df[df["cell_type"].isin(classes)].copy()
    df["x_px"] = (df["x_centroid"] / microns_per_px).round().astype(int)
    df["y_px"] = (df["y_centroid"] / microns_per_px).round().astype(int)
    out = df[["cell_id", "x_px", "y_px", "cell_type"]]
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CELL_DIR / f"{sample}.parquet")
    print(f"[{sample}] {len(out)} cells -> cells/{sample}.parquet", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", help="single sample; omit to process every xenium/*.parquet")
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes = cfg["classes"]
    microns_per_px = cfg["data"]["microns_per_px"]
    samples = [args.sample] if args.sample else sorted(p.stem for p in XENIUM_DIR.glob("*.parquet"))
    for s in samples:
        build(s, classes, microns_per_px)


if __name__ == "__main__":
    main()
