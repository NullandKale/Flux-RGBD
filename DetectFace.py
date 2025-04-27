#!/usr/bin/env python3
# DetectFace.py ─ YOLOv8-Face wrapper with GUI time-series integration
# --------------------------------------------------------------
# pip install ultralytics pillow requests
#
# Usage:
#   from DetectFace import FaceDetector
#   fd   = FaceDetector(device="cuda", gui=gui)  # or cpu, optional GUI
#   boxes, probs = fd.detect(rgb_numpy)         # per-frame
#   rgb_boxed    = fd.draw(rgb_numpy, boxes)    # optional
# --------------------------------------------------------------

from __future__ import annotations
import os
import requests
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple

# try to import GUI for time-series logging
try:
    from GUI import GUI
except ImportError:
    GUI = None

_MODEL_URL  = (
    "https://github.com/akanametov/yolov8-face/releases/download/"
    "v0.0.0/yolov8l-face.pt"
)
_MODEL_PATH = "yolov8l-face.pt"


def _ensure_model(path=_MODEL_PATH, url=_MODEL_URL) -> str:
    if os.path.exists(path):
        return path
    print("Downloading YOLOv8-Face weights…")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


class FaceDetector:
    """Light wrapper around yolov8-face that works on RGB numpy frames,
       and logs face-count into GUI if provided.
    """

    def __init__(
        self,
        device: str | None = None,
        conf: float = 0.25,
        gui: GUI | None = None
    ) -> None:
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(_ensure_model()).to(device)
        self.model.fuse()  # small speed gain
        self.device = device
        self.conf   = conf
        self.gui    = gui

    # ----------------------------------------------------------

    @torch.inference_mode()
    def detect(
        self, rgb: np.ndarray, margin: int = 10
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Parameters
        ----------
        rgb : H×W×3 uint8 RGB image
        margin : extra pixels to include around the detected box

        Returns
        -------
        boxes  : list[(x0,y0,x1,y1)]   (int, pixel coords)
        scores : list[float]           confidence 0-1
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("detect expects H×W×3 uint8 RGB frame")

        res = self.model.predict(
            source=rgb,
            imgsz=max(rgb.shape[:2]),
            conf=self.conf,
            verbose=False
        )[0]

        h, w = rgb.shape[:2]
        boxes, scores = [], []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            # add margin then clamp
            x0, y0, x1, y1 = (
                max(int(x0) - margin, 0),
                max(int(y0) - margin, 0),
                min(int(x1) + margin, w - 1),
                min(int(y1) + margin, h - 1),
            )
            boxes.append((x0, y0, x1, y1))
            scores.append(float(b.conf))

        # log face-count to GUI time-series
        if self.gui:
            count = len(boxes)
            # mode=0 for text display (current + 1s/5s avg)
            # assume up to 20 faces in view for scale
            self.gui.addTimeSeriesData(
                "face_count",
                float(count),
                min_val=0.0,
                max_val=20.0,
                mode=0
            )

            self.gui.addTimeSeriesData(
                "face_count_graph",
                float(count),
                min_val=0.0,
                max_val=20.0,
                mode=1
            )

        return boxes, scores

    # ----------------------------------------------------------

    @staticmethod
    def draw(
        rgb: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float] | None = None,
        color=(0, 255, 0),
    ) -> np.ndarray:
        """Return an RGB copy with rectangles (and optional scores) drawn."""
        vis = rgb.copy()
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            if scores is not None:
                txt = f"{scores[i]*100:.1f}%"
                cv2.putText(
                    vis,
                    txt,
                    (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return vis
