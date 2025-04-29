#!/usr/bin/env python3
# DepthFilter.py – enhanced temporal depth stabiliser with extended parameterization,
# full GUI logging, and decayed-history smoothing.
# ---------------------------------------------------------------------------------
from Frame import Frame, FrameHistory
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Optional

# Optional GUI interface
try:
    from GUI import GUI
except ImportError:
    GUI = None

class DepthFilter:
    """
    Temporal depth stabiliser with extended parameters, full GUI logging,
    and decayed-history smoothing, now using FrameHistory for frame management.

    Usage:
        filter_frame(frame: Frame) -> np.ndarray  # returns H×W float32
    """

    def __init__(
        self,
        device: Optional[str] = None,
        fps: float = 30.0,
        window: int = 9,
        history_decay: float = 0.4,
        keyframe_interval: int = 3,
        scene_thresh: float = 0.1,
        sharpen_strength: float = 0.1,
        var_beta: float = 7.5,
        grad_beta: float = 10.0,
        motion_kernel: int = 5,
        eq_alpha: float = 0.7,
        median_window: int = 3,
        gui: Optional[GUI] = None
    ):
        # — Core parameters —
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fps = fps
        self.window = max(1, window)
        self.history_decay = float(np.clip(history_decay, 0.0, 1.0))
        self.keyframe_interval = keyframe_interval
        self.scene_thresh = scene_thresh
        self.sharpen_strength = sharpen_strength
        self.var_beta = var_beta
        self.grad_beta = grad_beta
        self.motion_kernel = motion_kernel if motion_kernel % 2 == 1 else motion_kernel + 1
        self.eq_alpha = eq_alpha
        self.median_window = max(1, median_window if median_window % 2 == 1 else median_window + 1)
        self.gui = gui

        # — Frame history management —
        self.history = FrameHistory(maxlen=self.window)

        # — Rolling state (legacy) —
        self.prev_edges: Optional[torch.Tensor] = None
        self.prev_depth: Optional[torch.Tensor] = None
        self.keyframe_depth: Optional[torch.Tensor] = None
        self.prev_rgb: Optional[torch.Tensor] = None
        self.frame_count = 0
        self.keyframe_idx = 0

        # — Precompute Sobel kernels on device —
        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             dtype=torch.float32, device=self.device)
        self.sobel_x = sobel.view(1, 1, 3, 3)
        self.sobel_y = sobel.t().view(1, 1, 3, 3)

    def filter_frame(self, frame: Frame) -> np.ndarray:
        """
        Wraps the core filter, storing results in frame and appending to history.
        Requires frame.d_nat to be set (via frame.depth()).
        """
        if frame.d_nat is None:
            raise RuntimeError("Frame native depth not initialized. Call frame.depth() first.")

        # Apply existing filter logic
        filtered = self.filter(frame.d_nat, frame.rgb)

        # Store filtered depth in frame and append to history
        frame.d_filtered = filtered
        self.history.append(frame)

        return filtered

    def filter(self, d32: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """
        Main entry point.
        Applies scene-cut detection, temporal smoothing, variance/gradient blending,
        optional GUI logging, and returns filtered depth (H×W float32).
        """
        # Convert inputs to GPU tensors
        depth_t = torch.tensor(d32, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        rgb_t   = torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255.0

        # 1) Raw depth log
        if self.gui:
            self._log_buffer(depth_t, "raw_depth")

        # 2) Scene-cut detection and possible reset
        scene_flag = self._detect_scene_cut(rgb_t)
        if scene_flag or self.prev_depth is None:
            return self._reset_state(depth_t, rgb_t, scene_flag, original=d32)

        self.frame_count += 1

        # 3) Compute and log edge map
        curr_edges = self._compute_edges(depth_t)
        if self.gui:
            self._log_buffer(curr_edges, "edge_map")

        # 4) Update history buffers
        self._update_history(depth_t)

        # 5) Compute per-pixel weights
        var_w   = self._variance_weight()
        grad_w  = self._gradient_weight()
        motion  = self._motion_mask(var_w, grad_w)

        # 6) Align to reference and compute blend weight
        ref     = self._choose_reference()
        aligned = self._align(depth_t, ref)
        blend_w = self._compute_blend_weight(aligned)

        # 7) Compute decayed-history average
        history_avg = self._history_average()

        # 8) Fuse, apply inertia, and sharpen
        new_depth = self._fuse_and_sharpen(aligned, history_avg, blend_w)

        # 9) Final logging
        if self.gui:
            self._final_logging(new_depth, depth_t, scene_flag)

        # 10) Update rolling state
        self._update_state(new_depth, curr_edges, rgb_t)

        # Return as H×W float32 NumPy array
        return new_depth.squeeze().cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────────
    # Private Helpers — Core Computations
    # ──────────────────────────────────────────────────────────────────────────────

    def _compute_edges(self, depth: torch.Tensor) -> torch.Tensor:
        pad = F.pad(depth, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(pad, self.sobel_x)
        gy = F.conv2d(pad, self.sobel_y)
        return torch.sqrt(gx * gx + gy * gy)

    def _align(self, current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        cur = current.view(-1).double()
        refv = reference.view(-1).double()
        mc, mr = cur.mean(), refv.mean()
        cov = ((cur - mc) * (refv - mr)).sum() / max(cur.numel() - 1, 1)
        varc = ((cur - mc) ** 2).sum() / max(cur.numel() - 1, 1)
        if varc < 1e-12 or not torch.isfinite(cov):
            a, b = 1.0, 0.0
        else:
            a = (cov / varc).float()
            b = (mr - a.double() * mc).float()
        return a * current + b

    def _update_history(self, depth_t: torch.Tensor):
        self.prev_frames.append(depth_t)
        if len(self.prev_frames) > self.window:
            self.prev_frames.pop(0)

    def _variance_weight(self) -> torch.Tensor:
        stack   = torch.stack(self.prev_frames, dim=0)
        var_map = stack.var(dim=0, unbiased=False).squeeze()
        var_norm = (var_map / max(var_map.max(), 1e-6)).clamp(0, 1)
        w = (var_norm * self.var_beta).clamp(0, 1)
        if self.gui:
            self._log_buffer(var_norm.unsqueeze(0).unsqueeze(0), "variance_map")
            self.gui.addTimeSeriesData("var_mean", float(var_map.mean()), mode=1)
        return w

    def _gradient_weight(self) -> torch.Tensor:
        if len(self.prev_frames) < 2:
            return self._variance_weight()
        last, prev = self.prev_frames[-1].squeeze(), self.prev_frames[-2].squeeze()
        grad_norm = ((last - prev).abs() / max((last - prev).abs().max(), 1e-6)).clamp(0, 1)
        w = (grad_norm * self.grad_beta).clamp(0, 1)
        if self.gui:
            self._log_buffer(grad_norm.unsqueeze(0).unsqueeze(0), "gradient_map")
            self.gui.addTimeSeriesData("grad_mean", float(grad_norm.mean()), mode=1)
        return w

    def _motion_mask(self, var_w: torch.Tensor, grad_w: torch.Tensor) -> torch.Tensor:
        m = torch.maximum(var_w, grad_w)
        m = (m > 0.05).float() * m
        m = m.unsqueeze(0).unsqueeze(0)
        m = F.max_pool2d(m,
                         kernel_size=self.motion_kernel,
                         stride=1,
                         padding=self.motion_kernel // 2)
        if self.gui:
            self._log_buffer(m, "motion_map")
            self.gui.addTimeSeriesData("motion_mean", float(m.mean()), mode=1)
        return m

    def _choose_reference(self) -> torch.Tensor:
        if self.keyframe_interval and (self.frame_count - self.keyframe_idx) >= self.keyframe_interval:
            return self.keyframe_depth
        return self.prev_depth

    def _compute_blend_weight(self, aligned: torch.Tensor) -> torch.Tensor:
        ref    = self._choose_reference()                                  # [1,1,H,W]
        diff_r = ((aligned - ref).abs() / (ref.abs() + 1e-6)).clamp(0, 1)   # [1,1,H,W]
        if self.gui:
            self._log_buffer(diff_r, "diff_ratio")
        motion = self._motion_mask(self._variance_weight(), self._gradient_weight())
        return torch.maximum(diff_r, motion)                                # [1,1,H,W]

    def _history_average(self) -> torch.Tensor:
        N = len(self.prev_frames)
        frames = torch.stack(self.prev_frames, dim=0).view(N, 1, *self.prev_frames[0].shape[-2:])
        weights = torch.tensor(
            [self.history_decay ** (N - 1 - i) for i in range(N)],
            dtype=torch.float32,
            device=self.device
        ).view(N, 1, 1, 1)
        weights /= weights.sum()
        hist = (frames * weights).sum(dim=0, keepdim=True)
        if self.gui:
            self._log_buffer(hist, "history_avg")
        return hist

    def _fuse_and_sharpen(self, aligned: torch.Tensor, history: torch.Tensor, blend_w: torch.Tensor) -> torch.Tensor:
        fused = blend_w * aligned + (1 - blend_w) * self._choose_reference()
        base  = 0.5 * fused + 0.5 * history
        new_depth = self.eq_alpha * base + (1 - self.eq_alpha) * self._choose_reference()
        blur = F.avg_pool2d(new_depth, kernel_size=3, stride=1, padding=1)
        return (new_depth + self.sharpen_strength * (new_depth - blur)).clamp(min=0.0)

    # ──────────────────────────────────────────────────────────────────────────────
    # Private Helpers — State & Logging
    # ──────────────────────────────────────────────────────────────────────────────

    def _detect_scene_cut(self, rgb_t: torch.Tensor) -> float:
        if self.prev_rgb is None:
            return 0.0
        change = float((rgb_t - self.prev_rgb).abs().mean())
        if self.gui:
            self.gui.addTimeSeriesData("rgb_change", change, mode=1)
        return 1.0 if change > self.scene_thresh else 0.0

    def _reset_state(self, depth_t: torch.Tensor, rgb_t: torch.Tensor, scene_flag: float, original: np.ndarray) -> np.ndarray:
        self.prev_frames    = [depth_t.clone()]
        self.prev_edges     = self._compute_edges(depth_t)
        self.prev_depth     = depth_t.clone()
        self.keyframe_depth = depth_t.clone()
        self.frame_count    = 1
        self.keyframe_idx   = 0
        self.prev_rgb       = rgb_t.clone()
        if self.gui:
            self.gui.addTimeSeriesData("scene_cut", scene_flag, mode=1)
        return original.astype(np.float32)

    def _update_state(self, new_depth: torch.Tensor, edges: torch.Tensor, rgb_t: torch.Tensor):
        self.prev_depth   = new_depth.detach()
        self.prev_edges   = edges.detach()
        self.prev_rgb     = rgb_t.detach()
        if self.keyframe_interval and (self.frame_count - self.keyframe_idx) >= self.keyframe_interval:
            self.keyframe_depth = new_depth.clone()
            self.keyframe_idx   = self.frame_count

    def _log_buffer(self, tensor: torch.Tensor, name: str):
        """Convert a [1,1,H,W] or [1,H,W] tensor to BGR and log via GUI."""
        arr = tensor.squeeze().cpu().numpy()
        vis = ((arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        self.gui.addBuffer(name, bgr)

    def _final_logging(self, new_depth: torch.Tensor, depth_t: torch.Tensor, scene_flag: float):
        """Log final difference and scene-cut flag."""
        # difference between filtered and raw
        out_diff = (new_depth - depth_t).abs().squeeze()
        dmin, dmax = float(out_diff.min()), float(out_diff.max())
        norm2 = ((out_diff - dmin) / max(dmax - dmin, 1e-6) * 255).cpu().numpy().astype(np.uint8)
        self.gui.addBuffer("diff_output", cv2.cvtColor(norm2, cv2.COLOR_GRAY2BGR))
        # log scene-cut occurrence
        self.gui.addTimeSeriesData("scene_cut", scene_flag, mode=1)
