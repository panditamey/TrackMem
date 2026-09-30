"""TrackNet-style detection metrics.

Per frame (prediction visible iff heatmap max > threshold):
    TP   pred visible, GT visible, distance <= tolerance
    FP1  pred visible, GT visible, distance >  tolerance
    FP2  pred visible, GT invisible
    TN   pred invisible, GT invisible
    FN   pred invisible, GT visible
precision = TP / (TP + FP1 + FP2), recall = TP / (TP + FN), accuracy = (TP + TN) / all.
"""
import cv2
import numpy as np

ORIG_SCALE = 1280 / 512  # input space -> original 1280x720 space


def heatmap_to_point(hm, threshold):
    """Center of the bounding box of the largest thresholded blob (TrackNetV2/V3 convention).
    Returns (visible, x, y, peak) in heatmap pixel coordinates."""
    peak = float(hm.max())
    if peak <= threshold:
        return 0, np.nan, np.nan, peak
    mask = (hm > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return 1, x + (w - 1) / 2, y + (h - 1) / 2, peak


def classify(pred_vis, pred_xy, gt_vis, gt_xy, tolerance, scale=1.0):
    """Vectorized per-frame outcome labels. xy arrays in input space; distance scaled by `scale`."""
    dist = np.linalg.norm(pred_xy - gt_xy, axis=1) * scale
    out = np.empty(len(gt_vis), dtype=object)
    pv, gv = pred_vis.astype(bool), gt_vis.astype(bool)
    out[pv & gv & (dist <= tolerance)] = "TP"
    out[pv & gv & ~(dist <= tolerance)] = "FP1"
    out[pv & ~gv] = "FP2"
    out[~pv & ~gv] = "TN"
    out[~pv & gv] = "FN"
    return out, dist


def summarize(outcomes, dist=None, gt_vis=None):
    c = {k: int((outcomes == k).sum()) for k in ("TP", "FP1", "FP2", "TN", "FN")}
    n = len(outcomes)
    prec = c["TP"] / max(c["TP"] + c["FP1"] + c["FP2"], 1)
    rec = c["TP"] / max(c["TP"] + c["FN"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    res = {"n": n, **c, "precision": prec, "recall": rec, "f1": f1,
           "accuracy": (c["TP"] + c["TN"]) / max(n, 1)}
    if dist is not None and gt_vis is not None:
        d = dist[(gt_vis == 1) & np.isfinite(dist)]
        res["loc_err_median"] = float(np.median(d)) if len(d) else float("nan")
        res["loc_err_p90"] = float(np.percentile(d, 90)) if len(d) else float("nan")
    return res
