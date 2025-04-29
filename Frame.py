# Frame.py

from collections import deque
import numpy as np
from typing import Deque, Optional, Tuple, List, Dict

class Frame:
    """
    Holds:
      - raw RGB
      - depth bytes → native depth
      - face-boxes
      - filtered / autofocus / text-filtered depths
      - visualization buffers & metrics
    """
    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self.depth_bytes: Optional[bytes] = None
        self.d_nat: Optional[np.ndarray] = None
        self.boxes: List[Tuple[int,int,int,int]] = []
        self.d_filtered: Optional[np.ndarray] = None
        self.d_autofocus: Optional[np.ndarray] = None
        self.d_text: Optional[np.ndarray] = None
        self.buffers: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, float] = {}

    def depth(self, depth_model, width: int, height: int) -> np.ndarray:
        if self.d_nat is None:
            self.depth_bytes = depth_model.process(self.rgb, width, height)
            self.d_nat = np.frombuffer(self.depth_bytes, np.float32).reshape(height, width)
        return self.d_nat

    def depth_vis(self) -> np.ndarray:
        """
        Convert the current depth map into a BGR grayscale visualization.
        Chooses in order: d_text, d_autofocus, d_filtered, then d_nat.
        """
        if 'vis' not in self.buffers:
            # pick the highest‐priority available depth map
            if self.d_text is not None:
                d = self.d_text
            elif self.d_autofocus is not None:
                d = self.d_autofocus
            elif self.d_filtered is not None:
                d = self.d_filtered
            else:
                d = self.d_nat

            mn, mx = float(d.min()), float(d.max())
            if mx - mn < 1e-6:
                mx = mn + 1e-6
            g = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
            self.buffers['vis'] = np.repeat(g[..., None], 3, axis=2)
        return self.buffers['vis']

class FrameHistory:
    """Keeps the last N Frame objects for temporal passes (e.g. filtering)."""
    def __init__(self, maxlen: int):
        self._queue: Deque[Frame] = deque(maxlen=maxlen)

    def append(self, frame: Frame):
        self._queue.append(frame)

    def latest(self, n: int) -> List[Frame]:
        return list(self._queue)[-n:]
