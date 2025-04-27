#!/usr/bin/env python3
# DepthAutoFocus_fixedrange.py
# -------------------------------------------------------------------------------
# As before, but uses a fixed normalization range from the first call.

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

try:
    from GUI import GUI
except ImportError:
    GUI = None

class DepthAutoFocus:
    """
    Face-aware depth shift with a fixed normalization range.
    """

    def __init__(
        self,
        fps: float,
        target: float = 0.5,
        epsilon: float = 0.03,
        strength: float = 1.0,
        momentum: float = 0.15,
        max_step: float = 1.0,
        gui: Optional[GUI] = None,
    ):
        # ... same parameter checks as before ...
        self.fps      = fps
        self.target   = target
        self.epsilon  = epsilon
        self.strength = strength
        self.gui      = gui

        # per-frame params
        self._momentum_frame = momentum ** (1.0 / fps)
        self._max_step_frame = max_step  / fps

        # state
        self._prev_shift_raw   = 0.0
        self._prev_desired_raw = 0.0

        # **new**: fixed range for normalization
        self._base_dmin = None
        self._base_dmax = None

    def _velocity_clamp(self, desired: float) -> float:
        delta = desired - self._prev_shift_raw
        if self._max_step_frame > 0:
            delta = np.clip(delta, -self._max_step_frame, self._max_step_frame)
        return self._prev_shift_raw + delta

    def remap(
        self,
        depth: np.ndarray,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        # 0) On first call, capture base min/max
        if self._base_dmin is None:
            self._base_dmin = float(depth.min())
            self._base_dmax = float(depth.max())

        base_range = max(1e-6, self._base_dmax - self._base_dmin)

        # 1) Apply existing cumulative shift
        shifted = depth + self._prev_shift_raw

        # 2) If no faces, decay back to zero and log
        if not boxes:
            decay = -np.sign(self._prev_shift_raw) * min(
                abs(self._prev_shift_raw),
                self._max_step_frame
            )
            self._prev_shift_raw += decay

            if self.gui:
                self.gui.addTimeSeriesData(
                    "af_applied_norm",
                    self._prev_shift_raw / base_range,
                    min_val=-1.0, max_val=1.0, mode=1
                )
            return shifted

        # 3) Build face mask on shifted depth
        h, w = shifted.shape
        mask = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in boxes:
            x0, x1 = max(0, x0), min(w-1, x1)
            y0, y1 = max(0, y0), min(h-1, y1)
            if x1>=x0 and y1>=y0:
                mask[y0:y1+1, x0:x1+1] = True
        if not mask.any():
            return shifted

        # 4) Normalize with **fixed** base range
        depth_norm = (shifted - self._base_dmin) / base_range

        # 5) Median face depth & error
        face_med = float(np.median(depth_norm[mask]))
        err      = self.target - face_med

        # log
        if self.gui:
            self.gui.addTimeSeriesData("af_face_med", face_med,
                min_val=0.0, max_val=1.0, mode=0)
            self.gui.addTimeSeriesData("af_error", err,
                min_val=-1.0, max_val=1.0, mode=0)

        # 6) Dead-zone & desired_norm
        if abs(err) < self.epsilon:
            desired_norm = 0.0
        else:
            residual     = np.sign(err) * (abs(err) - self.epsilon)
            desired_norm = self.strength * residual

        if self.gui:
            self.gui.addTimeSeriesData("af_desired_norm", desired_norm,
                min_val=-1.0, max_val=1.0, mode=1)

        # 7) Convert to raw, momentum, clamp, update shift
        desired_raw = desired_norm * base_range
        desired_raw = (self._momentum_frame * self._prev_desired_raw
                       + (1-self._momentum_frame) * desired_raw)
        self._prev_desired_raw = desired_raw

        applied = self._velocity_clamp(desired_raw)
        self._prev_shift_raw = applied

        if self.gui:
            self.gui.addTimeSeriesData("af_applied_norm", applied/base_range,
                min_val=-1.0, max_val=1.0, mode=1)

        # 8) Return depth + updated shift
        return depth + applied
