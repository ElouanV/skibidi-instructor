"""Reference building (from pro/coach footage) and comparison of a skier against it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import TURN_FEATURES, estimate_view, frame_metrics, segment_turns, summarize_turns
from .knowledge import FAULTS, METRICS
from .pose import PoseSequence

PCTS = (10, 25, 50, 75, 90)
REF_DIR = Path(__file__).resolve().parent.parent / "references"
DISCIPLINES = ("carving", "freeride", "park")


def analyze_sequence(seq: PoseSequence) -> dict:
    m = frame_metrics(seq)
    turns = segment_turns(m, seq.fps)
    return {"metrics": m, "turns": turns, "summary": summarize_turns(turns), "view": estimate_view(m)}


# --------------------------------------------------------------------------- reference


def build_reference(pose_files: list[str | Path], discipline: str, name: str = "",
                    require_frontal: bool = True) -> dict:
    """Aggregate per-turn feature distributions over many pro/coach clips."""
    values: dict[str, list[float]] = {f: [] for f in TURN_FEATURES}
    used, skipped = [], []
    for p in pose_files:
        res = analyze_sequence(PoseSequence.load(p))
        if require_frontal and res["view"]["view"] not in ("frontal", "oblique"):
            skipped.append({"file": str(p), "reason": f"view={res['view']['view']}"})
            continue
        if not res["turns"]:
            skipped.append({"file": str(p), "reason": "no turns detected"})
            continue
        used.append({"file": str(p), "n_turns": len(res["turns"])})
        for t in res["turns"]:
            for f in TURN_FEATURES:
                if np.isfinite(t.features[f]):
                    values[f].append(t.features[f])
    ref = {
        "name": name or f"{discipline} reference",
        "discipline": discipline,
        "source": "pose-derived",
        "clips": used,
        "skipped": skipped,
        "features": {
            f: {f"p{q}": float(np.percentile(v, q)) for q in PCTS} | {"n": len(v)}
            for f, v in values.items() if len(v) >= 5
        },
    }
    return ref


def load_reference(discipline: str, path: str | Path | None = None) -> dict:
    path = Path(path) if path else REF_DIR / f"{discipline}_prior.json"
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- comparison


def _percentile_position(x: float, dist: dict) -> float:
    """Approximate percentile of x within a reference distribution described by p10..p90."""
    qs = [q for q in PCTS if f"p{q}" in dist]
    vs = [dist[f"p{q}"] for q in qs]
    if x <= vs[0]:
        return float(qs[0]) * (x / vs[0]) if vs[0] > 0 and x > 0 else 0.0
    if x >= vs[-1]:
        return 100.0
    return float(np.interp(x, vs, qs))


def compare(summary: dict, ref: dict) -> list[dict]:
    out = []
    for f, dist in ref["features"].items():
        x = summary["overall"].get(f)
        if x is None:
            continue
        pos = _percentile_position(x, dist)
        status = "below_reference" if x < dist["p10"] else "above_reference" if x > dist["p90"] else "within_reference"
        out.append({
            "metric": f,
            "label": METRICS[f]["label"],
            "unit": METRICS[f]["unit"],
            "value": round(x, 3),
            "reference_median": round(dist["p50"], 3),
            "reference_range_p10_p90": [round(dist["p10"], 3), round(dist["p90"], 3)],
            "percentile_vs_reference": round(pos, 1),
            "status": status,
        })
    return out


# Rules: (fault id, disciplines, predicate on overall summary, severity fn)
def _v(s, k):
    return s["overall"].get(k)


def detect_faults(summary: dict, discipline: str, view: dict) -> list[dict]:
    s = summary
    faults = []

    def add(fid, severity, evidence, plane="general"):
        conf = {"lateral": view["lateral_metric_confidence"],
                "sagittal": view["sagittal_metric_confidence"],
                "general": max(view["lateral_metric_confidence"], view["sagittal_metric_confidence"])}[plane]
        faults.append({"id": fid, **{k: FAULTS[fid][k] for k in ("title", "why", "drills")},
                       "severity": round(float(np.clip(severity, 0, 1)), 2), "confidence": round(conf, 2),
                       "evidence": evidence})

    incl, ang = _v(s, "peak_inclination"), _v(s, "apex_angulation")
    if incl is not None and ang is not None and incl > 15 and ang < 8:
        add("banking", (8 - ang) / 12 + 0.3, f"peak inclination {incl:.0f}° with only {ang:.0f}° angulation", "lateral")

    rng = _v(s, "extension_range")
    if rng is not None and rng < 0.08:
        add("static_stance", (0.08 - rng) / 0.08 + 0.2, f"flexion–extension range {rng:.2f} (pros ≈ 0.15+)", "sagittal")

    knee = _v(s, "apex_knee_flex")
    if knee is not None and knee < 30:
        add("too_tall", (30 - knee) / 30 + 0.2, f"knee flexion at apex {knee:.0f}°", "sagittal")

    drop = _v(s, "apex_shoulder_drop")
    if drop is not None and drop > 5:
        add("inside_shoulder_drop", (drop - 5) / 10 + 0.3, f"inside shoulder {drop:.0f}° low at apex", "lateral")

    kar = _v(s, "apex_knee_ankle_ratio")
    if kar is not None and kar < 0.8:
        add("a_frame", (0.8 - kar) / 0.4 + 0.3, f"knee gap is {kar:.2f}× the ankle gap at apex", "lateral")

    sw = _v(s, "apex_stance_width")
    if discipline == "carving" and sw is not None and sw < 0.3:
        add("narrow_stance", (0.3 - sw) / 0.3 + 0.2, f"stance width {sw:.2f}× torso", "lateral")

    hh = _v(s, "mean_hand_height")
    if hh is not None and hh < 0:
        add("hands_low", -hh / 0.3 + 0.2, f"hands {abs(hh):.2f}× torso below hips on average")

    hs = _v(s, "mean_hand_spread")
    if hs is not None and hs < 1.1:
        add("hands_narrow", (1.1 - hs) / 0.5 + 0.2, f"hand spread {hs:.2f}× shoulder width", "lateral")

    if discipline == "carving" and incl is not None and incl < 20:
        add("low_edge_angle", (20 - incl) / 20 + 0.2, f"peak inclination {incl:.0f}°", "lateral")

    asym = s["asymmetry"]
    for k, thr in (("peak_inclination", 10), ("apex_angulation", 8)):
        a = asym.get(k)
        if a is not None and abs(a) > thr:
            weaker = "right" if a > 0 else "left"
            add("side_asymmetry", abs(a) / (3 * thr) + 0.3,
                f"{METRICS[k]['label']} differs by {abs(a):.0f}° between sides; image-{weaker} turns are weaker", "lateral")
            break

    return sorted(faults, key=lambda f: -f["severity"] * f["confidence"])


def build_report_data(res: dict, ref: dict, discipline: str, skier_level: str | None = None) -> dict:
    view = res["view"]
    return {
        "discipline": discipline,
        "skier_level": skier_level,
        "camera_view": view,
        "n_turns": res["summary"]["n_turns"],
        "turn_summary": res["summary"],
        "reference": {"name": ref["name"], "source": ref.get("source"), "n_clips": len(ref.get("clips", []))},
        "comparison": compare(res["summary"], ref),
        "faults": detect_faults(res["summary"], discipline, view),
        "metric_glossary": {k: {kk: v[kk] for kk in ("label", "plain", "technical")} for k, v in METRICS.items()},
        "caveats": _caveats(res, view),
    }


def _caveats(res: dict, view: dict) -> list[str]:
    c = ["Measurements come from 2D video; angles are projections and depend on camera position."]
    if view["view"] == "side":
        c.append("Footage looks side-on: lateral metrics (lean, angulation, stance width) are unreliable. "
                 "Film from below or behind, along the fall line.")
    elif view["view"] == "oblique":
        c.append("Camera angle is oblique; lateral metrics are approximate.")
    if view["view"] == "frontal":
        c.append("Filmed head-on: knee/hip flexion are foreshortened and read lower than reality. Compare them "
                 "only with references filmed the same way, or add a side-on clip.")
    if res["summary"]["n_turns"] < 4:
        c.append(f"Only {res['summary']['n_turns']} turns detected; film at least 6–8 turns for stable results.")
    return c
