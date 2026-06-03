#!/usr/bin/env python
"""LoRA fine-tuning of the backbone on labeled cell crops.

Standard parameter-efficient fine-tuning: LoRA adapters are attached to the
backbone, a linear head is trained on the Xenium-derived labels, and the adapter
is saved. extract_features.py --lora then extracts features from the adapted
backbone. The fine-tuning here is a generic reference, not a tuned recipe.

Output: $HECELL_ROOT/outputs/lora/   (a PEFT adapter)
"""
import argparse

import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from backbone import load_backbone
from config import CELL_DIR, LORA_DIR, load_config
from crops import CellCrops, read_he


def collect(samples, classes):
    """Per sample, return (image, xs, ys, label_idx) for the labeled cells."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    out = []
    for s in samples:
        cpath = CELL_DIR / f"{s}.parquet"
        if not cpath.exists():
            continue
        cells = pd.read_parquet(cpath)
        cells = cells[cells["cell_type"].isin(class_to_idx)]
        if not len(cells):
            continue
        out.append((read_he(s), cells["x_px"].to_numpy(), cells["y_px"].to_numpy(),
                    cells["cell_type"].map(class_to_idx).to_numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--groups", required=True, help="CSV with a 'sample' column")
    ap.add_argument("--backbone", default="vit_base_patch16_224")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["classes"]
    lc = cfg.get("lora", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = load_backbone(args.backbone, args.weights, device)
    model = get_peft_model(base, LoraConfig(
        r=lc.get("r", 8), lora_alpha=lc.get("alpha", 16), lora_dropout=lc.get("dropout", 0.1),
        target_modules=lc.get("target_modules", ["qkv", "proj"]))).to(device)
    head = nn.Linear(base.num_features, len(classes)).to(device)

    data = collect(pd.read_csv(args.groups)["sample"].tolist(), classes)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
                            lr=lc.get("lr", 1e-4))
    loss_fn = nn.CrossEntropyLoss()
    model.train(); head.train()
    for epoch in range(lc.get("epochs", 3)):
        for image, xs, ys, labels in data:
            loader = DataLoader(CellCrops(image, xs, ys, labels),
                                batch_size=lc.get("batch_size", 128), shuffle=True)
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(head(model(xb)), yb).backward()
                opt.step()
        print(f"epoch {epoch} done", flush=True)

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_DIR))
    print(f"saved LoRA adapter -> {LORA_DIR}", flush=True)


if __name__ == "__main__":
    main()
