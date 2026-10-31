"""Visual outputs: skeleton overlay video and metric timeline plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pose import SKELETON, PoseSequence


def overlay_video(video_path: str | Path, seq: PoseSequence, metrics: dict, turns: list, out_path: str | Path,
                  stride: int = 1) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (w, h))
    apexes = {t.apex for t in turns}
    i = f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= seq.n_frames:
            break
        if i % stride:
            i += 1
            continue
        k = seq.kpts[f]
        for a, b in SKELETON:
            if np.isfinite(k[[a, b]]).all():
                cv2.line(frame, tuple(map(int, k[a])), tuple(map(int, k[b])), (0, 255, 255), 3)
        for p in k:
            if np.isfinite(p).all():
                cv2.circle(frame, tuple(map(int, p)), 4, (0, 0, 255), -1)
        txt = [f"incl {metrics['inclination'][f]:+.0f}  ang {metrics['angulation'][f]:+.0f}",
               f"knee {metrics['knee_flex'][f]:.0f}  hip {metrics['hip_flex'][f]:.0f}"]
        for j, t in enumerate(txt):
            cv2.putText(frame, t, (20, 40 + 35 * j), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if f in apexes:
            cv2.putText(frame, "APEX", (20, 40 + 35 * 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3, cv2.LINE_AA)
        writer.write(frame)
        i += 1
        f += 1
    cap.release()
    writer.release()


def timeline_plot(metrics: dict, turns: list, fps: float, out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(metrics["inclination"])) / fps
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    panels = [(["inclination", "angulation"], "degrees"), (["knee_flex", "hip_flex"], "degrees"),
              (["extension"], "hip–ankle / leg length")]
    for ax, (names, ylabel) in zip(axes, panels):
        for n in names:
            ax.plot(t, metrics[n], label=n.replace("_", " "), linewidth=1.6)
        for tr in turns:
            ax.axvline(tr.apex / fps, color="#999", linewidth=0.6, linestyle=":")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("time (s) — dotted lines = turn apex")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
