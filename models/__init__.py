from .trackmem import TrackMem
from .tracknetv5 import TrackNetV5

MODELS = ("tracknetv5", "trackmem")

# Checkpoints saved before the TrackNetV5 port (model name "b0") were TrackNetV5 without R-STR
# using Conv -> BN -> ReLU, nearest upsampling and a biased head. Kept so they still load.
_LEGACY = {"b0": dict(rstr=False, conv_order="conv_bn_relu", upsample="nearest", head_bias=True,
                    mdd_eps=1e-3)}


def build_model(cfg):
    m, d = cfg["model"], cfg["data"]
    name = m["name"]
    if name in _LEGACY:
        return TrackNetV5(d["seq_len"], (d["height"], d["width"]), width_mult=m.get("width_mult", 1.0),
                          **_LEGACY[name])
    if name == "tracknetv5":
        return TrackNetV5(d["seq_len"], (d["height"], d["width"]), **m.get("tracknetv5", {}))
    if name == "trackmem":
        return TrackMem(d["seq_len"], (d["height"], d["width"]), **m.get("trackmem", {}))
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
