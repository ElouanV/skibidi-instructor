"""Skeleton extraction: video -> per-frame 2D keypoints of the main skier.

Uses a YOLO pose model (COCO 17-keypoint layout). In each frame we keep a
single person: the one closest to the previously tracked skier, falling back
to the largest confident detection. This handles the usual ski footage case
(one filmed skier, other people occasionally crossing the frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

SKELETON = [
    (L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
]

DEFAULT_MODEL = "yolo11m-pose.pt"


@dataclass
class PoseSequence:
    """Keypoints of one skier across a video. Missing frames are NaN."""

    kpts: np.ndarray  # (T, 17, 2) pixel coordinates
    conf: np.ndarray  # (T, 17) keypoint confidence
    boxes: np.ndarray  # (T, 4) xyxy
    fps: float
    width: int
    height: int
    source: str = ""

    @property
    def n_frames(self) -> int:
        return self.kpts.shape[0]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, kpts=self.kpts, conf=self.conf, boxes=self.boxes,
            meta=np.array([self.fps, self.width, self.height]),
            source=np.array(self.source),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PoseSequence":
        d = np.load(path, allow_pickle=False)
        fps, w, h = d["meta"]
        return cls(d["kpts"], d["conf"], d["boxes"], float(fps), int(w), int(h), str(d["source"]))


def _pick_person(boxes: np.ndarray, confs: np.ndarray, prev_center: np.ndarray | None) -> int | None:
    if len(boxes) == 0:
        return None
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if prev_center is not None:
        diag = np.sqrt(areas.max())
        dist = np.linalg.norm(centers - prev_center, axis=1)
        close = dist < 1.5 * diag
        if close.any():
            idx = np.where(close)[0]
            return int(idx[np.argmin(dist[idx])])
    return int(np.argmax(areas * confs))


def extract(video_path: str | Path, model_name: str = DEFAULT_MODEL, stride: int = 1,
            device: str | None = None, progress: bool = True) -> PoseSequence:
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts, conf, boxes = [], [], []
    prev_center = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride:
            i += 1
            continue
        res = model(frame, verbose=False, device=device)[0]
        k = np.full((17, 2), np.nan)
        c = np.zeros(17)
        b = np.full(4, np.nan)
        if res.keypoints is not None and len(res.boxes):
            bx = res.boxes.xyxy.cpu().numpy()
            bc = res.boxes.conf.cpu().numpy()
            j = _pick_person(bx, bc, prev_center)
            if j is not None:
                k = res.keypoints.xy[j].cpu().numpy().astype(float)
                c = (res.keypoints.conf[j].cpu().numpy() if res.keypoints.conf is not None
                     else np.ones(17))
                b = bx[j]
                prev_center = (b[:2] + b[2:]) / 2
        kpts.append(k)
        conf.append(c)
        boxes.append(b)
        if progress and total and len(kpts) % 50 == 0:
            print(f"  pose: frame {i}/{total}", flush=True)
        i += 1
    cap.release()

    seq = PoseSequence(np.array(kpts), np.array(conf), np.array(boxes),
                       fps / stride, width, height, str(video_path))
    # Low-confidence keypoints are treated as missing.
    seq.kpts[seq.conf < 0.3] = np.nan
    return seq
