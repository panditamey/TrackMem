"""Per-frame evaluation subsets derived from GT labels only (so they are model independent).

Tags (a frame can have several):
    visible / invisible
    occl_short     invisible run <= short_gap, bracketed by visible, brackets not near top edge
    occl_long      same but run > short_gap
    out_of_frame   bracketed run whose bracket points are near the top edge (e.g. clears)
    edge_invisible invisible run at start/end of rally
    reappear       first `reappear_frames` visible frames after a bracketed run >= reappear_min_gap
    fast           visible, speed > fast_speed px/frame (input space)
    hit            within hit_window frames of a velocity direction change > hit_angle
    normal         visible and none of: fast, hit, reappear, adjacent to an invisible frame
"""
import numpy as np


def invisible_runs(vis):
    runs, i, n = [], 0, len(vis)
    while i < n:
        if vis[i] == 0:
            j = i
            while j < n and vis[j] == 0:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def frame_tags(vis, xy, height, cfg):
    s = cfg["subsets"]
    n = len(vis)
    v = vis.astype(bool)
    tags = {k: np.zeros(n, bool) for k in
            ("occl_short", "occl_long", "out_of_frame", "edge_invisible", "reappear", "fast", "hit")}
    tags["visible"], tags["invisible"] = v.copy(), ~v

    for i, j in invisible_runs(vis):
        if i == 0 or j == n:
            tags["edge_invisible"][i:j] = True
            continue
        near_top = min(xy[i - 1, 1], xy[j, 1]) < s["top_border"] * height
        if near_top:
            tags["out_of_frame"][i:j] = True
        elif j - i <= s["short_gap"]:
            tags["occl_short"][i:j] = True
        else:
            tags["occl_long"][i:j] = True
        if j - i >= s["reappear_min_gap"]:
            k = j
            while k < n and k < j + s["reappear_frames"] and v[k]:
                tags["reappear"][k] = True
                k += 1

    step = np.full(n, np.nan)                      # displacement into frame t from t-1
    ok = v[1:] & v[:-1]
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    step[1:][ok] = d[ok]
    speed = np.fmax(step, np.append(step[1:], np.nan))
    tags["fast"] = v & (np.nan_to_num(speed) > s["fast_speed"])

    for t in range(1, n - 1):
        if v[t - 1] and v[t] and v[t + 1]:
            a, b = xy[t] - xy[t - 1], xy[t + 1] - xy[t]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1 and nb > 1:
                ang = np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))
                if ang > s["hit_angle"]:
                    w = s["hit_window"]
                    tags["hit"][max(0, t - w):t + w + 1] = True
    tags["hit"] &= v

    adj_invisible = np.zeros(n, bool)
    adj_invisible[1:] |= ~v[:-1]
    adj_invisible[:-1] |= ~v[1:]
    tags["normal"] = v & ~tags["fast"] & ~tags["hit"] & ~tags["reappear"] & ~adj_invisible
    return tags
