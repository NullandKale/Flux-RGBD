#!/usr/bin/env python3
# DepthAutoFocus.py – GPU-accelerated autofocus via PyTorch

from __future__ import annotations
from typing import Optional
import numpy as np
import torch

from Frame import Frame

try:
    from GUI import GUI
except ImportError:
    GUI = None

class DepthAutoFocus:
    """
    Face-aware depth shift with a fixed normalization range,
    fully on GPU via PyTorch.
    """

    def __init__(
        self,
        fps: float,
        target: float = 0.5,
        epsilon: float = 0.005,
        strength: float = 4.0,
        momentum: float = 0.05,
        max_step: float = 4.0,
        gui: Optional[GUI] = None,
        device: Optional[str] = None,
    ):
        self.fps      = fps
        self.target   = target
        self.epsilon  = epsilon
        self.strength = strength
        self.gui      = gui

        # select device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # per-frame factors on GPU
        self._momentum_frame = torch.tensor(momentum ** (1.0 / fps), device=self.device)
        self._max_step_frame = torch.tensor(max_step   / fps,      device=self.device)

        # will be initialized on first call
        self._base_dmin  = None  # torch scalar
        self._base_range = None  # torch scalar

        # stateful shifts as torch scalars
        self._prev_shift_raw   = torch.tensor(0.0, device=self.device)
        self._prev_desired_raw = torch.tensor(0.0, device=self.device)

        # zero scalar for dead-zone
        self._zero = torch.tensor(0.0, device=self.device)

    def remap(self, frame: Frame) -> np.ndarray:
        # 1) pull depth into GPU tensor
        depth_np = frame.d_filtered if frame.d_filtered is not None else frame.d_nat
        depth_t  = torch.from_numpy(depth_np).to(self.device).float()

        # 2) init normalization range once
        if self._base_range is None:
            dmin = depth_t.min()
            dmax = depth_t.max()
            self._base_dmin  = dmin
            self._base_range = torch.clamp(dmax - dmin, min=torch.tensor(1e-6, device=self.device))

        # 3) ensure persistent GPU buffer
        if not hasattr(frame, "d_autofocus_t") or frame.d_autofocus_t.shape != depth_t.shape:
            frame.d_autofocus_t = torch.empty_like(depth_t)

        # 4) apply previous shift globally
        frame.d_autofocus_t = depth_t + self._prev_shift_raw

        # 5) if no faces, just decay shift
        if not frame.boxes:
            decay = -torch.sign(self._prev_shift_raw) * torch.min(
                torch.abs(self._prev_shift_raw), self._max_step_frame
            )
            self._prev_shift_raw = self._prev_shift_raw + decay
            result_t = frame.d_autofocus_t
        else:
            # 6) build mask for face ROIs
            mask = torch.zeros_like(depth_t, dtype=torch.bool)
            for x0, y0, x1, y1 in frame.boxes:
                x0c, x1c = max(0, x0), min(depth_t.shape[1], x1)
                y0c, y1c = max(0, y0), min(depth_t.shape[0], y1)
                mask[y0c:y1c, x0c:x1c] = True

            vals = frame.d_autofocus_t[mask]
            if vals.numel() == 0:
                result_t = frame.d_autofocus_t
            else:
                # 7) compute face-region mean on GPU
                face_mean = vals.mean()
                norm_med  = (face_mean - self._base_dmin) / self._base_range
                err       = self.target - norm_med

                # 8) dead-zone + strength -> desired normalized shift
                desired_norm = torch.where(
                    torch.abs(err) < self.epsilon,
                    self._zero,
                    self.strength * (torch.sign(err) * (torch.abs(err) - self.epsilon))
                )

                # 9) momentum & velocity clamp
                desired_raw = (
                    self._momentum_frame * self._prev_desired_raw
                    + (1 - self._momentum_frame) * (desired_norm * self._base_range)
                )
                self._prev_desired_raw = desired_raw

                delta   = desired_raw - self._prev_shift_raw
                delta   = torch.clamp(delta, -self._max_step_frame, self._max_step_frame)
                applied = self._prev_shift_raw + delta
                self._prev_shift_raw = applied

                # 10) final global shift
                result_t = depth_t + applied

                # merged logging: difference between applied and desired (normalized)
                if self.gui:
                    applied_norm = applied / self._base_range
                    diff_norm = applied_norm - desired_norm
                    self.gui.addTimeSeriesData("af_diff", diff_norm.item(), mode=1)

        # 11) bring result back to CPU numpy
        frame.d_autofocus = result_t.cpu().numpy()
        return frame.d_autofocus
