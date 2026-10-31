"""End-to-end tests on a synthetic skier (no video or model needed)."""

import numpy as np
import pytest

from skibidi.analysis import analyze_sequence, build_report_data, build_reference, detect_faults, load_reference
from skibidi.coach import template_report
from skibidi.pose import (L_ANK, L_ELB, L_HIP, L_KNEE, L_SHO, L_WRI, R_ANK, R_ELB, R_HIP, R_KNEE, R_SHO, R_WRI,
                          PoseSequence)


def synthetic_skier(lean_amp=40.0, upper_ratio=0.5, flex_amp=0.15, knee_bend=50.0, n_turns=6, fps=30,
                    turn_s=1.2, stance=40.0, knee_pinch=1.0):
    """Frontal-view skier: legs lean by lean(t), upper body leans by upper_ratio*lean(t)."""
    T = int(n_turns * turn_s * fps)
    t = np.arange(T) / fps
    lean = lean_amp * np.sin(np.pi * t / turn_s)
    k = np.full((T, 17, 2), np.nan)
    thigh = shank = 45.0
    torso = 60.0
    for i in range(T):
        lr = np.radians(lean[i])
        up = np.array([np.sin(lr), -np.cos(lr)])
        side = np.array([np.cos(lr), np.sin(lr)])
        bend = knee_bend * (1 - flex_amp * 3 * np.cos(2 * np.pi * t[i] / turn_s))
        leg = np.sqrt(thigh**2 + shank**2 + 2 * thigh * shank * np.cos(np.radians(bend)))
        ank_mid = np.array([300.0, 400.0])
        hip_mid = ank_mid + up * leg
        for (a, kn, h, s) in ((L_ANK, L_KNEE, L_HIP, -1), (R_ANK, R_KNEE, R_HIP, 1)):
            k[i, a] = ank_mid + side * s * stance / 2
            k[i, h] = hip_mid + side * s * 15
            mid = (k[i, a] + k[i, h]) / 2
            # knees bend towards the camera (not visible laterally); pinch < 1 moves them inwards
            k[i, kn] = mid + side * s * (stance / 2 * (knee_pinch - 1))
        ur = np.radians(lean[i] * upper_ratio)
        uup = np.array([np.sin(ur), -np.cos(ur)])
        uside = np.array([np.cos(ur), np.sin(ur)])
        sho_mid = hip_mid + uup * torso
        k[i, L_SHO], k[i, R_SHO] = sho_mid - uside * 22, sho_mid + uside * 22
        k[i, L_ELB], k[i, R_ELB] = k[i, L_SHO] + [-12, 30], k[i, R_SHO] + [12, 30]
        k[i, L_WRI], k[i, R_WRI] = hip_mid - uside * 40 - uup * 15, hip_mid + uside * 40 - uup * 15
    conf = np.ones((T, 17))
    return PoseSequence(k, conf, np.zeros((T, 4)), fps, 640, 480, "synthetic")


def test_turns_detected_and_alternate():
    res = analyze_sequence(synthetic_skier())
    turns = res["turns"]
    assert 5 <= len(turns) <= 6
    dirs = [t.direction for t in turns]
    assert all(a != b for a, b in zip(dirs, dirs[1:]))
    assert res["view"]["view"] == "frontal"


def test_angulation_tracks_upper_body():
    good = analyze_sequence(synthetic_skier(upper_ratio=0.4))["summary"]["overall"]
    bank = analyze_sequence(synthetic_skier(upper_ratio=1.0))["summary"]["overall"]
    assert good["apex_angulation"] > 15
    assert abs(bank["apex_angulation"]) < 3
    assert good["peak_inclination"] < bank["peak_inclination"]


def test_banking_fault_detected():
    res = analyze_sequence(synthetic_skier(upper_ratio=1.0))
    ids = [f["id"] for f in detect_faults(res["summary"], "carving", res["view"])]
    assert "banking" in ids


def test_a_frame_detected():
    res = analyze_sequence(synthetic_skier(knee_pinch=0.5))
    ids = [f["id"] for f in detect_faults(res["summary"], "carving", res["view"])]
    assert "a_frame" in ids


def test_reference_and_report(tmp_path):
    files = []
    for i, ratio in enumerate((0.3, 0.4, 0.5)):
        p = tmp_path / f"pro{i}.npz"
        synthetic_skier(upper_ratio=ratio).save(p)
        files.append(p)
    ref = build_reference(files, "carving", "test pros")
    assert len(ref["clips"]) == 3
    assert "apex_angulation" in ref["features"]
    res = analyze_sequence(PoseSequence.load(files[0]))
    data = build_report_data(res, ref, "carving")
    md = template_report(data)
    assert "Technical appendix" in md


@pytest.mark.parametrize("d", ["carving", "freeride", "park"])
def test_priors_load(d):
    ref = load_reference(d)
    assert ref["features"]
