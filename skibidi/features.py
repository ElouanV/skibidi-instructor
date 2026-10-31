"""Biomechanical features from 2D keypoints.

Conventions (image coordinates, y points down):
* Lean angles are measured from the image vertical, in degrees. Positive means
  the segment leans towards image-right.
* Lateral metrics (lean, angulation, shoulder drop, stance width) are only
  meaningful when the skier is filmed roughly along the fall line (from below
  or behind). `estimate_view` flags side-on footage so the analysis can lower
  its confidence instead of giving wrong advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pose import (L_ANK, L_HIP, L_KNEE, L_SHO, L_WRI, R_ANK, R_HIP, R_KNEE, R_SHO, R_WRI,
                   PoseSequence)


def _nan_smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average that ignores NaNs; short gaps are interpolated."""
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x.copy()
    filled = np.interp(idx, idx[ok], x[ok])
    # Don't invent data across long gaps (> 1s at 30fps).
    gap = np.zeros(len(x), bool)
    run = 0
    for i, v in enumerate(ok):
        run = 0 if v else run + 1
        if run > 30:
            gap[i - run + 1:i + 1] = True
    if win > 1:
        k = np.ones(win) / win
        pad = win // 2
        filled = np.convolve(np.pad(filled, pad, mode="edge"), k, mode="valid")[:len(x)]
    filled[gap] = np.nan
    return filled


def _angle_at(a, b, c) -> np.ndarray:
    """Interior angle ABC in degrees, vectorised over frames."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _lean(lower, upper) -> np.ndarray:
    v = upper - lower
    return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))


def _line_tilt(p, q) -> np.ndarray:
    """Tilt of the line through p,q vs horizontal; positive if the image-right end is higher."""
    left = np.where((p[..., 0] <= q[..., 0])[..., None], p, q)
    right = np.where((p[..., 0] <= q[..., 0])[..., None], q, p)
    d = right - left
    return np.degrees(np.arctan2(-d[..., 1], d[..., 0]))


def frame_metrics(seq: PoseSequence) -> dict[str, np.ndarray]:
    k = seq.kpts
    sho = (k[:, L_SHO] + k[:, R_SHO]) / 2
    hip = (k[:, L_HIP] + k[:, R_HIP]) / 2
    knee = (k[:, L_KNEE] + k[:, R_KNEE]) / 2
    ank = (k[:, L_ANK] + k[:, R_ANK]) / 2

    torso = np.linalg.norm(sho - hip, axis=1)
    torso_ref = np.nanmedian(torso) if np.isfinite(torso).any() else np.nan

    knee_l = 180 - _angle_at(k[:, L_HIP], k[:, L_KNEE], k[:, L_ANK])
    knee_r = 180 - _angle_at(k[:, R_HIP], k[:, R_KNEE], k[:, R_ANK])
    hip_flex = 180 - _angle_at(sho, hip, knee)

    lower_lean = _lean(ank, hip)
    upper_lean = _lean(hip, sho)
    turn_sign = np.sign(lower_lean)
    angulation = (lower_lean - upper_lean) * turn_sign

    shoulder_tilt = _line_tilt(k[:, L_SHO], k[:, R_SHO])
    hip_tilt = _line_tilt(k[:, L_HIP], k[:, R_HIP])
    # A shoulder line square to the torso tilts by -upper_lean. Anything beyond that is a lateral
    # side-bend of the trunk; positive = inside shoulder dipped towards the snow.
    shoulder_drop = (-shoulder_tilt - upper_lean) * turn_sign

    ankle_gap = np.abs(k[:, L_ANK, 0] - k[:, R_ANK, 0])
    knee_gap = np.abs(k[:, L_KNEE, 0] - k[:, R_KNEE, 0])
    sho_w = np.linalg.norm(k[:, L_SHO] - k[:, R_SHO], axis=1)

    leg_len = (np.linalg.norm(k[:, L_HIP] - k[:, L_KNEE], axis=1) + np.linalg.norm(k[:, L_KNEE] - k[:, L_ANK], axis=1)
               + np.linalg.norm(k[:, R_HIP] - k[:, R_KNEE], axis=1) + np.linalg.norm(k[:, R_KNEE] - k[:, R_ANK], axis=1)) / 2
    extension = np.linalg.norm(hip - ank, axis=1) / leg_len

    wri = (k[:, L_WRI] + k[:, R_WRI]) / 2
    hand_height = (hip[:, 1] - wri[:, 1]) / torso_ref
    hand_spread = np.abs(k[:, L_WRI, 0] - k[:, R_WRI, 0]) / sho_w

    with np.errstate(invalid="ignore", divide="ignore"):
        m = {
            "knee_flex_l": knee_l,
            "knee_flex_r": knee_r,
            "knee_flex": (knee_l + knee_r) / 2,
            "hip_flex": hip_flex,
            "lower_lean": lower_lean,
            "upper_lean": upper_lean,
            "inclination": _lean(ank, sho),
            "angulation": angulation,
            "shoulder_drop": shoulder_drop,
            "hip_tilt": hip_tilt,
            "stance_width": ankle_gap / torso_ref,
            "knee_ankle_ratio": knee_gap / ankle_gap,
            "extension": extension,
            "hand_height": hand_height,
            "hand_spread": hand_spread,
            "shoulder_width_ratio": sho_w / torso,
        }
    win = max(3, int(round(seq.fps / 10)) | 1)
    return {name: _nan_smooth(v, win) for name, v in m.items()}


def estimate_view(metrics: dict[str, np.ndarray]) -> dict:
    """Rough camera-view classification from apparent shoulder width / torso length."""
    r = float(np.nanmedian(metrics["shoulder_width_ratio"]))
    if np.isnan(r):
        view = "unknown"
    elif r > 0.55:
        view = "frontal"  # filmed from below/behind, along the fall line
    elif r > 0.3:
        view = "oblique"
    else:
        view = "side"
    lateral = {"frontal": 1.0, "oblique": 0.6, "side": 0.25, "unknown": 0.0}[view]
    # Knee/hip flexion happen in the sagittal plane: foreshortened when filmed head-on.
    sagittal = {"frontal": 0.4, "oblique": 0.7, "side": 1.0, "unknown": 0.0}[view]
    return {"view": view, "shoulder_width_ratio": r, "lateral_metric_confidence": lateral,
            "sagittal_metric_confidence": sagittal}


@dataclass
class Turn:
    start: int
    end: int
    apex: int
    direction: str  # "right" or "left" (image space)
    features: dict[str, float] = field(default_factory=dict)


TURN_FEATURES = [
    "peak_inclination", "apex_angulation", "apex_knee_flex", "apex_hip_flex",
    "apex_shoulder_drop", "apex_stance_width", "apex_knee_ankle_ratio",
    "extension_range", "mean_hand_height", "mean_hand_spread", "duration_s",
]


def segment_turns(metrics: dict[str, np.ndarray], fps: float, min_peak_deg: float = 8.0,
                  min_duration_s: float = 0.35) -> list[Turn]:
    """Turns = contiguous runs where the legs lean to one side, with a clear apex."""
    lean = metrics["lower_lean"]
    sign = np.sign(np.nan_to_num(lean))
    turns: list[Turn] = []
    i, n = 0, len(lean)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and sign[j + 1] == sign[i]:
            j += 1
        seg = np.abs(lean[i:j + 1])
        if (j - i + 1) / fps >= min_duration_s and np.nanmax(seg) >= min_peak_deg:
            apex = i + int(np.nanargmax(seg))
            t = Turn(i, j, apex, "right" if sign[i] > 0 else "left")
            t.features = _turn_features(metrics, t, fps)
            turns.append(t)
        i = j + 1
    return turns


def _turn_features(m: dict[str, np.ndarray], t: Turn, fps: float) -> dict[str, float]:
    s = slice(t.start, t.end + 1)
    a = t.apex
    ext = m["extension"][s]
    f = {
        "peak_inclination": abs(m["inclination"][a]),
        "apex_angulation": m["angulation"][a],
        "apex_knee_flex": m["knee_flex"][a],
        "apex_hip_flex": m["hip_flex"][a],
        "apex_shoulder_drop": m["shoulder_drop"][a],
        "apex_stance_width": m["stance_width"][a],
        "apex_knee_ankle_ratio": m["knee_ankle_ratio"][a],
        "extension_range": np.nanmax(ext) - np.nanmin(ext) if np.isfinite(ext).any() else np.nan,
        "mean_hand_height": np.nanmean(m["hand_height"][s]),
        "mean_hand_spread": np.nanmean(m["hand_spread"][s]),
        "duration_s": (t.end - t.start + 1) / fps,
    }
    return {k: float(v) for k, v in f.items()}


def summarize_turns(turns: list[Turn]) -> dict:
    """Median of each turn feature, overall and per side, plus left/right asymmetry."""
    out: dict = {"n_turns": len(turns), "overall": {}, "left": {}, "right": {}, "asymmetry": {}}
    for name in TURN_FEATURES:
        for side in ("overall", "left", "right"):
            vals = [t.features[name] for t in turns if side == "overall" or t.direction == side]
            vals = [v for v in vals if np.isfinite(v)]
            out[side][name] = float(np.median(vals)) if vals else None
        l, r = out["left"][name], out["right"][name]
        out["asymmetry"][name] = (l - r) if (l is not None and r is not None) else None
    return out
