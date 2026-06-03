"""Configuration and path resolution.

Code and config live in the repository (REPO). Data and derived artifacts live
under HECELL_ROOT, which defaults to the repository root but can point elsewhere.
"""
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HECELL_ROOT", REPO))

DATA = ROOT / "data"
OUT = ROOT / "outputs"

XENIUM_DIR = DATA / "xenium"
CELL_DIR = DATA / "cells"
HE_DIR = DATA / "he_aligned"
FEATURE_DIR = OUT / "features"
LORA_DIR = OUT / "lora"
MODEL_DIR = OUT / "models"
PRED_DIR = OUT / "predictions"
FIG_DIR = OUT / "figures"

DEFAULT_CONFIG = REPO / "configs" / "baseline.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)
