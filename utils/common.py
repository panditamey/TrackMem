import random

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    """Load YAML config; overrides are 'a.b.c=value' strings parsed as YAML scalars."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides):
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = _parse(value)
    return cfg


def _parse(value):
    """YAML scalar, but also accept numbers YAML 1.1 reads as strings (e.g. '3e-4')."""
    v = yaml.safe_load(value)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def amp_dtype(mode, device):
    """Resolve AMP mode to a dtype (None = AMP off)."""
    if device.type != "cuda" or mode == "off":
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]
