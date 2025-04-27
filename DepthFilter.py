#!/usr/bin/env python3
# DepthFilter.py – temporal depth stabiliser with fps-aware smoothing, scene-cut detection, and GUI logging
# ---------------------------------------------------------------------------------------------
# filter(depth32) → float32 H×W
#
# This class now performs:
#   • **scene-cut detection** via frame-to-frame mean-abs-difference
#   • edge / variance adaptive blending (fps-aware)
#   • short-window median fusion
#   • un-sharp masking for detail
#   • FPS-aware variance smoothing
#   • Logs scene-cut events to GUI (0 or 1)
# ---------------------------------------------------------------------------------------------

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

# Optional GUI interface
try:
    from GUI import GUI
except ImportError:
    GUI = None

class DepthFilter:
    """
    Temporal depth stabiliser with optional GUI logging.

    Call signature:
        filter(d32: np.ndarray) -> np.ndarray
    """

    def __init__(
        self,
        device: Optional[str] = None,
        fps: float = 30.0,
        window: int = 32,
        keyframe_interval: int = 0,
        scene_thresh: float = 0.1,
        sharpen_strength: float = 0.1,
        var_beta: float = 10.0,
        gui: GUI | None = None
    ):
        """
        :param fps:               Frames per second.
        :param window:            Rolling history size for variance.
        :param keyframe_interval: Frames between forced keyframes.
        :param scene_thresh:      Mean-abs-diff threshold [0..1] for scene cuts.
        :param sharpen_strength:  Unsharp-mask strength.
        :param var_beta:          Time constant (seconds) for variance smoothing.
        :param gui:               Optional GUI to log data.
        """
        self.gui = gui
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fps = fps
        self.dt  = 1.0 / fps
        self.window = window
        self.keyframe_interval = keyframe_interval
        self.scene_thresh = scene_thresh
        self.sharpen_strength = sharpen_strength
        self.var_beta = var_beta

        # rolling state
        self.frame_count    = 0
        self.prev_depth     = None   # Tensor [1×1×H×W]
        self.prev_edges     = None
        self.prev_rgb       = None   # Tensor [1×3×H×W] for scene-cut on color
        self.keyframe_depth = None
        self.keyframe_idx   = 0
        self.prev_frames    = []     # list of raw-depth Tensors

        # Sobel kernels for edge detection
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)

    def filter(self, d32: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        d32 : H×W float32 raw depth map
        rgb : H×W×3 uint8 or float32 color frame

        Returns
        -------
        np.ndarray H×W filtered depth
        """
        scene_flag = 0.0

        # prepare tensors
        depth_t = torch.as_tensor(d32, dtype=torch.float32, device=self.device)\
                         .unsqueeze(0).unsqueeze(0)
        # normalize rgb to [0..1], shape [1×3×H×W]
        rgb_t = (torch.as_tensor(rgb, dtype=torch.float32, device=self.device)
                    .permute(2,0,1).unsqueeze(0) / 255.0)

        # scene-cut detection on color
        if self.prev_rgb is not None:
            diff_rgb = (rgb_t - self.prev_rgb).abs()
            mean_diff = float(diff_rgb.mean())
            if mean_diff > self.scene_thresh:
                scene_flag = 1.0
                # reset all state on scene cut
                pad = F.pad(depth_t, (1,1,1,1), mode="replicate")
                gx = F.conv2d(pad, self.sobel_x)
                gy = F.conv2d(pad, self.sobel_y)
                self.prev_edges     = torch.sqrt(gx*gx + gy*gy)
                self.prev_depth     = depth_t.clone()
                self.keyframe_depth = depth_t.clone()
                self.prev_frames    = [depth_t]
                self.frame_count    = 1
                self.keyframe_idx   = 0
                # log scene cut
                if self.gui:
                    self.gui.addTimeSeriesData(
                        "scene_cut", scene_flag,
                        min_val=0.0, max_val=1.0, mode=1
                    )
                # update rgb state and bail
                self.prev_rgb = rgb_t.clone()
                return d32
        else:
            # first-frame init
            pad = F.pad(depth_t, (1,1,1,1), mode="replicate")
            gx = F.conv2d(pad, self.sobel_x)
            gy = F.conv2d(pad, self.sobel_y)
            self.prev_edges     = torch.sqrt(gx*gx + gy*gy)
            self.prev_depth     = depth_t.clone()
            self.keyframe_depth = depth_t.clone()
            self.prev_frames    = [depth_t]
            self.frame_count    = 1
            # log no scene cut
            if self.gui:
                self.gui.addTimeSeriesData(
                    "scene_cut", scene_flag,
                    min_val=0.0, max_val=1.0, mode=1
                )
            self.prev_rgb = rgb_t.clone()
            return d32

        self.frame_count += 1

        # edge maps
        pad = F.pad(depth_t, (1,1,1,1), mode="replicate")
        gx = F.conv2d(pad, self.sobel_x)
        gy = F.conv2d(pad, self.sobel_y)
        curr_edges = torch.sqrt(gx*gx + gy*gy)

        # reference frame
        ref = self.keyframe_depth if self.keyframe_interval else self.prev_depth

        # affine alignment
        prev_flat = ref.view(-1).double()
        curr_flat = depth_t.view(-1).double()
        N = curr_flat.numel()
        m_prev, m_curr = prev_flat.mean(), curr_flat.mean()
        cov = torch.dot(curr_flat - m_curr, prev_flat - m_prev) / max(N-1,1)
        var_c = ((curr_flat - m_curr)**2).sum() / max(N-1,1)
        if var_c.abs() < 1e-12 or not torch.isfinite(cov):
            a, b = 1.0, 0.0
        else:
            a = (cov / var_c).float()
            b = (m_prev - a.double()*m_curr).float()
        aligned = a * depth_t + b

        # diff-based weight
        diff = (aligned - ref).abs()
        diff_r = diff / (ref.abs() + 1e-6)
        ramp = (diff_r - 0.05).clamp(0.0, 0.15) / 0.15
        w = torch.where(diff_r >= 0.20, torch.ones_like(diff), ramp)
        edge_mask = (self.prev_edges > 1e-3) | (curr_edges > 1e-3)
        w = torch.where(edge_mask, torch.ones_like(w), w)

        # variance-aware bump
        self.prev_frames.append(depth_t)
        if len(self.prev_frames) > self.window:
            self.prev_frames.pop(0)
        var_hist = torch.stack(self.prev_frames,0).var(0, unbiased=False)
        adapt_w = var_hist / (var_hist + 1e-6)
        per_sec = self.dt / self.var_beta
        w = torch.maximum(w, adapt_w * per_sec)

        # blend & post
        blended = w * aligned + (1.0 - w) * ref
        if len(self.prev_frames) >= 3:
            med = torch.median(torch.stack(self.prev_frames[-3:],0),0)[0]
            blended = 0.5 * blended + 0.5 * med
        blur = F.avg_pool2d(blended, 3, 1, 1)
        sharpened = blended + self.sharpen_strength * (blended - blur)
        new_depth = sharpened.clamp(min=0.0)

        # update state
        self.prev_depth = new_depth.detach()
        self.prev_edges = curr_edges.detach()
        self.prev_rgb   = rgb_t.detach()
        if self.keyframe_interval and (self.frame_count - self.keyframe_idx) >= self.keyframe_interval:
            self.keyframe_depth = new_depth.clone()
            self.keyframe_idx   = self.frame_count

        # log scene-flag
        if self.gui:
            self.gui.addTimeSeriesData(
                "scene_cut", scene_flag,
                min_val=0.0, max_val=1.0, mode=1
            )

        return new_depth.squeeze(0).squeeze(0).cpu().numpy()