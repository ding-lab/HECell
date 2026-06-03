#!/usr/bin/env python
"""Per-cell image features for the H&E cell-type baseline.

For each cell, a fixed window is cropped around its centroid on the aligned H&E
and encoded with a pathology foundation model, frozen by default, or with a
LoRA adapter from train_lora.py via --lora. One feature file is written per
sample; the trainer reads them back.

Inputs (per sample <s>):
  $HECELL_ROOT/data/he_aligned/<s>.tif      aligned H&E (HWC; multi-page tolerated)
  $HECELL_ROOT/data/cells/<s>.parquet       columns: cell_id, x_px, y_px, cell_type

Output:
  $HECELL_ROOT/outputs/features/<s>.pt       {"cell_id": ndarray, "feat": FloatTensor}

The backbone weights are not bundled. Pass --backbone (any timm model name) and
optionally --weights /path/to/checkpoint, or set FM_WEIGHTS; any public pathology
ViT backbone can be used.
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from backbone import load_backbone
from config import CELL_DIR, FEATURE_DIR as OUT_DIR
from crops import CellCrops, read_he


@torch.no_grad()
def encode(model, loader, device):
    out = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feat = model(batch)
        out.append(feat.float().cpu())
    return torch.cat(out)


def run(sample, model, device, batch, workers):
    cells = pd.read_parquet(CELL_DIR / f"{sample}.parquet")
    image = read_he(sample)
    ds = CellCrops(image, cells["x_px"].to_numpy(), cells["y_px"].to_numpy())
    loader = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True)
    feat = encode(model, loader, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"cell_id": cells["cell_id"].to_numpy(), "feat": feat}, OUT_DIR / f"{sample}.pt")
    print(f"[{sample}] {len(cells)} cells -> {tuple(feat.shape)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", help="single sample; omit to process every cells/*.parquet")
    ap.add_argument("--backbone", default="vit_base_patch16_224")
    ap.add_argument("--weights", default=os.environ.get("FM_WEIGHTS"))
    ap.add_argument("--lora", default=None, help="optional LoRA adapter dir from train_lora.py")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone(args.backbone, args.weights, device)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    samples = [args.sample] if args.sample else sorted(p.stem for p in CELL_DIR.glob("*.parquet"))
    for s in samples:
        run(s, model, device, args.batch, args.workers)


if __name__ == "__main__":
    main()
