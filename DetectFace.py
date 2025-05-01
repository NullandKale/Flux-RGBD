#!/usr/bin/env python3
# DetectFace.py ─ YOLOv11-Face wrapper supporting nano|small|large variants 
#                + auto-download + direct GPU uploads + resizing + GUI logging

import os
import requests
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from typing import List, Tuple

# Optional GUI interface
try:
    from GUI import GUI
except ImportError:
    GUI = None

from Frame import Frame

_MODEL_VARIANTS = {
    'nano':  'yolov11n-face.pt',
    'small': 'yolov11s-face.pt',
    'large': 'yolov11l-face.pt',
}
_BASE_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0"


class FaceDetector:
    """
    Wrapper around YOLOv11-Face that:
     • Supports model_size ∈ {'nano','small','large'}
     • Auto-downloads weights if missing
     • Requires CUDA
     • Uploads frames directly from NumPy to GPU (one copy)
     • Resizes to a multiple of 32 (model stride) to avoid warnings
     • Logs face-count via GUI.addTimeSeriesData(mode=0/1)
    """

    def __init__(
        self,
        model_size: str = 'nano',
        conf: float = 0.25,
        gui: GUI | None = None
    ) -> None:
        if model_size not in _MODEL_VARIANTS:
            raise ValueError(f"model_size must be one of {_MODEL_VARIANTS.keys()}")
        self.model_file = _MODEL_VARIANTS[model_size]
        self._ensure_model()

        assert torch.cuda.is_available(), "CUDA is required"
        self.device = torch.device("cuda:0")
        self.conf   = conf
        self.gui    = gui

        self.model = YOLO(self.model_file).to(self.device)
        self.model.fuse()

    def _ensure_model(self) -> None:
        if os.path.exists(self.model_file):
            return
        url = f"{_BASE_URL}/{self.model_file}"
        print(f"Downloading {self.model_file} from {url} …")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(self.model_file, "wb") as f:
            f.write(r.content)

    @torch.inference_mode()
    def detect(
        self,
        frame: Frame
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """
        Runs YOLOv11-Face on frame.rgb:
         1) Converts BGR→RGB and makes a contiguous copy
         2) Uploads directly to GPU (one copy)
         3) Resizes to a multiple of 32 (model stride) to avoid warnings
         4) Inference
         5) Scales back to original resolution
        Returns:
          boxes  : list of (x0,y0,x1,y1)
          scores : list of confidences
        """
        rgb = frame.rgb  # BGR H×W×3 uint8
        h, w = rgb.shape[:2]

        # 1) BGR→RGB contiguous copy
        rgb_rgb = rgb[:, :, ::-1].copy()  # H×W×3

        # 2) HWC→CHW, upload to GPU, normalize
        img_gpu = (
            torch.from_numpy(rgb_rgb)
            .permute(2, 0, 1)               # 3×H×W
            .to(self.device)                # one synchronous copy
            .float()
            .div_(255.0)
            .unsqueeze(0)                   # 1×3×H×W
        )

        # 3) Resize to square multiple of 32
        size = ((max(h, w) + 31) // 32) * 32
        img_resized = F.interpolate(
            img_gpu,
            size=(size, size),
            mode='bilinear',
            align_corners=False
        )

        # 4) Inference
        res = self.model.predict(
            source=img_resized,
            conf=self.conf,
            verbose=False
        )[0]

        # 5) Scale detections back
        sx = w / size
        sy = h / size
        boxes: List[Tuple[int,int,int,int]] = []
        scores: List[float] = []
        for b in res.boxes:
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            x0 *= sx; x1 *= sx
            y0 *= sy; y1 *= sy
            x0 = max(0, min(int(x0), w - 1))
            y0 = max(0, min(int(y0), h - 1))
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            boxes.append((x0, y0, x1, y1))
            scores.append(float(b.conf))

        # Attach & log
        frame.boxes  = boxes
        frame.scores = scores
        if self.gui:
            cnt = float(len(boxes))
            self.gui.addTimeSeriesData("face_count",       cnt, min_val=0.0, max_val=20.0, mode=0)
            self.gui.addTimeSeriesData("face_count_graph", cnt, min_val=0.0, max_val=20.0, mode=1)

        return boxes, scores

    @staticmethod
    def draw(
        rgb: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float] | None = None,
        color=(0, 255, 0),
    ) -> np.ndarray:
        vis = rgb.copy()
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            if scores:
                txt = f"{scores[i] * 100:.1f}%"
                cv2.putText(vis, txt, (x0, y0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return vis
