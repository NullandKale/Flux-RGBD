#!/usr/bin/env python3
# DepthFilter.py – enhanced temporal depth stabiliser with extended parameterization,
# full GUI logging, and decayed-history smoothing with median fusion.
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
    Temporal depth stabiliser with full GUI logging,
    storing all raw float32 buffers on Frame
    and re-adding short-window median fusion.
    """

    def __init__(
        self,
        device: Optional[str]    = None,
        fps: float               = 30.0,
        window: int              = 9,
        history_decay: float     = 0.4,
        keyframe_interval: int   = 3,
        scene_thresh: float      = 0.1,
        sharpen_strength: float  = 0.1,
        var_beta: float          = 20.0,
        grad_beta: float         = 10.0,
        motion_kernel: int       = 5,
        median_window: int       = 3,
        gui: Optional[GUI]       = None
    ):
        # — core params —
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fps               = fps
        self.window            = max(1, window)
        self.history_decay     = float(np.clip(history_decay, 0.0, 1.0))
        self.keyframe_interval = keyframe_interval
        self.scene_thresh      = scene_thresh
        self.sharpen_strength  = sharpen_strength
        self.var_beta          = var_beta
        self.grad_beta         = grad_beta
        self.motion_kernel     = motion_kernel if motion_kernel % 2 == 1 else motion_kernel + 1
        self.median_window     = max(1, median_window)
        self.gui               = gui

        # — history management —
        self.history = FrameHistory(maxlen=self.window)

        # — rolling state —
        self.prev_frames    = []   # list of raw-depth Tensors
        self.prev_edges     = None
        self.prev_depth     = None
        self.keyframe_depth = None
        self.prev_rgb       = None
        self.frame_count    = 0
        self.keyframe_idx   = 0

        # — precompute sobel kernels on device —
        sobel = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                             dtype=torch.float32, device=self.device)
        self.sobel_x = sobel.view(1,1,3,3)
        self.sobel_y = sobel.t().view(1,1,3,3)

    def filter_frame(self, frame: Frame) -> np.ndarray:
        # --- Stage 1: Load raw depth ---
        if frame.d_nat is None:
            raise RuntimeError("Call frame.depth(...) first")
        frame.raw_depth = frame.d_nat.astype(np.float32)

        # --- Stage 2: Pack tensors ---
        depth_t = torch.from_numpy(frame.raw_depth).to(self.device).unsqueeze(0).unsqueeze(0)
        rgb_t   = torch.from_numpy(frame.rgb).float().to(self.device)
        rgb_t   = rgb_t.permute(2,0,1).unsqueeze(0) / 255.0

        # --- Stage 3: Scene-cut & RGB change logging ---
        if self.prev_rgb is None:
            scene_flag = 0.0
            rgb_change = 0.0
        else:
            diff_rgb   = (rgb_t - self.prev_rgb).abs()
            rgb_change = float(diff_rgb.mean())
            scene_flag = 1.0 if rgb_change > self.scene_thresh else 0.0

        frame.scene_cut = scene_flag
        if self.gui:
            self.gui.addTimeSeriesData("1:scene_cut", scene_flag, mode=1)
            self.gui.addTimeSeriesData("1:rgb_change", rgb_change, mode=1)

        # --- Stage 4: Reset on scene-cut or first frame ---
        if scene_flag or self.prev_depth is None:
            out = self._reset_state(depth_t, rgb_t, frame.raw_depth)
            frame.d_filtered = out
            self.history.append(frame)
            return out
        self.frame_count += 1

        # --- Stage 5: Edge map ---
        curr_edges = self._compute_edges(depth_t)
        frame.edge_map = curr_edges.squeeze().cpu().numpy().astype(np.float32)
        if self.gui:
            self._log_buffer(curr_edges, "2:edge_map")

        # --- Stage 6: Update history ---
        self._update_history(depth_t)

        # --- Stage 7: Variance weight ---
        var_w = self._variance_weight(frame)

        # --- Stage 8: Gradient weight ---
        grad_w = self._gradient_weight(frame)

        # --- Stage 9: Motion mask ---
        motion = self._motion_mask(var_w, grad_w)

        # --- Stage 10: Affine align ---
        aligned = self._align(depth_t, self._choose_reference())

        # --- Stage 11: Diff ratio & blend weight ---
        diff_r = ((aligned - self._choose_reference()).abs() /
                  (self._choose_reference().abs() + 1e-6)).clamp(0,1)
        if self.gui:
            self._log_buffer(diff_r, "3:diff_ratio")
        bw = torch.maximum(diff_r, motion)
        frame.blend_weight = bw.squeeze().cpu().numpy().astype(np.float32)
        if self.gui:
            self._log_buffer(bw, "3:blend_weight")

        # --- Stage 12: Decayed-history average (logged only) ---
        if self.gui:
            N      = len(self.prev_frames)
            frames = torch.stack(self.prev_frames, dim=0)  # [N,1,H,W]
            weights= torch.tensor(
                [self.history_decay**(N-1-i) for i in range(N)],
                dtype=torch.float32, device=self.device
            ).view(N,1,1,1)
            weights /= weights.sum()

            # result is shape [1, H, W]
            hist = (frames * weights).sum(dim=0)
            hist2d = hist[0]  # remove the leading 1 → [H, W]
            frame.history_avg = hist2d.cpu().numpy().astype(np.float32)
            self._log_buffer(hist2d, "4:history_avg")

        # --- Stage 13: Median fusion & sharpen ---
        blend = bw * aligned + (1 - bw) * self._choose_reference()
        if len(self.prev_frames) >= self.median_window:
            med = torch.median(
                       torch.stack(self.prev_frames[-self.median_window:], 0),
                       dim=0
                   )[0]  # → shape [1,H,W]
            blend = 0.5 * blend + 0.5 * med
        blur = F.avg_pool2d(blend, kernel_size=3, stride=1, padding=1)
        newd = (blend + self.sharpen_strength * (blend - blur)).clamp(min=0.0)
        out  = newd.squeeze().cpu().numpy().astype(np.float32)

        # --- Stage 14: Optionally log keyframe depth ---
        if self.gui and self.keyframe_depth is not None:
            kf2d = self.keyframe_depth.squeeze(0)  # [H, W]
            self._log_buffer(kf2d, "5:keyframe_depth")

        # --- Stage 15: diff_output (raw abs diff) ---
        diff_tensor      = (newd - depth_t).abs().squeeze().cpu().numpy().astype(np.float32)
        frame.diff_output = diff_tensor
        if self.gui:
            maxd = diff_tensor.max() or 1e-6
            vis  = (diff_tensor / maxd * 255).astype(np.uint8)
            self.gui.addBuffer("6:diff_output", cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR))

        # --- Stage 16: Save & state update ---
        frame.d_filtered = out
        self._update_state(newd, curr_edges, rgb_t)
        self.history.append(frame)
        return out

    # ──────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────────

    def _compute_edges(self, depth: torch.Tensor) -> torch.Tensor:
        p  = F.pad(depth, (1,1,1,1), mode="replicate")
        gx = F.conv2d(p, self.sobel_x)
        gy = F.conv2d(p, self.sobel_y)
        return torch.sqrt(gx*gx + gy*gy)

    def _align(self, current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        c  = current.view(-1).double()
        r  = reference.view(-1).double()
        mc, mr = c.mean(), r.mean()
        cov    = ((c-mc)*(r-mr)).sum() / max(c.numel()-1,1)
        varc   = ((c-mc)**2).sum() / max(c.numel()-1,1)
        if varc < 1e-12 or not torch.isfinite(cov):
            a, b = 1.0, 0.0
        else:
            a = (cov/varc).float()
            b = (mr - a.double()*mc).float()
        return a*current + b

    def _update_history(self, depth_t: torch.Tensor):
        self.prev_frames.append(depth_t.detach())
        if len(self.prev_frames) > self.window:
            self.prev_frames.pop(0)

    def _variance_weight(self, frame: Frame) -> torch.Tensor:
        stack   = torch.stack(self.prev_frames, dim=0)
        var_map = stack.var(dim=0, unbiased=False).squeeze()
        vn      = (var_map / max(var_map.max(),1e-6)).clamp(0,1)
        vw      = (vn * self.var_beta).clamp(0,1)
        frame.variance_map = vn.cpu().numpy().astype(np.float32)
        frame.var_weight   = vw.cpu().numpy().astype(np.float32)
        if self.gui:
            self._log_buffer(vn.unsqueeze(0).unsqueeze(0), "7:variance_map")
            self.gui.addTimeSeriesData("7:var_mean", float(var_map.mean()), mode=1)
        return vw

    def _gradient_weight(self, frame: Frame) -> torch.Tensor:
        if len(self.prev_frames) < 2:
            return self._variance_weight(frame)
        last, prev = self.prev_frames[-1].squeeze(), self.prev_frames[-2].squeeze()
        gn = ((last - prev).abs() / max((last - prev).abs().max(),1e-6)).clamp(0,1)
        gw = (gn * self.grad_beta).clamp(0,1)
        frame.gradient_map = gn.cpu().numpy().astype(np.float32)
        frame.grad_weight  = gw.cpu().numpy().astype(np.float32)
        if self.gui:
            self._log_buffer(gn.unsqueeze(0).unsqueeze(0), "8:gradient_map")
            self.gui.addTimeSeriesData("8:grad_mean", float(gn.mean()), mode=1)
        return gw

    def _motion_mask(self, var_w: torch.Tensor, grad_w: torch.Tensor) -> torch.Tensor:
        m = torch.maximum(var_w, grad_w)
        m = (m>0.05).float() * m
        m = m.unsqueeze(0).unsqueeze(0)
        m = F.max_pool2d(m, kernel_size=self.motion_kernel, stride=1,
                        padding=self.motion_kernel//2)
        if self.gui:
            self._log_buffer(m.squeeze(0), "9:motion_map")
            self.gui.addTimeSeriesData("9:motion_mean", float(m.mean()), mode=1)
        return m

    def _choose_reference(self) -> torch.Tensor:
        if self.keyframe_interval and (self.frame_count - self.keyframe_idx) >= self.keyframe_interval:
            return self.keyframe_depth
        return self.prev_depth

    def _reset_state(self, depth_t: torch.Tensor, rgb_t: torch.Tensor, original: np.ndarray) -> np.ndarray:
        self.prev_frames    = [depth_t.clone()]
        self.prev_edges     = self._compute_edges(depth_t)
        self.prev_depth     = depth_t.clone()
        self.keyframe_depth = depth_t.clone()
        self.frame_count    = 1
        self.keyframe_idx   = 0
        self.prev_rgb       = rgb_t.clone()
        if self.gui:
            self.gui.addTimeSeriesData("1:scene_cut", 1.0, mode=1)
            kf2d = self.keyframe_depth.squeeze(0)
            self._log_buffer(kf2d, "5:keyframe_depth")
        return original.astype(np.float32)

    def _update_state(self, new_depth: torch.Tensor, edges: torch.Tensor, rgb_t: torch.Tensor):
        self.prev_depth = new_depth.detach()
        self.prev_edges = edges.detach()
        self.prev_rgb   = rgb_t.detach()
        if self.keyframe_interval and (self.frame_count - self.keyframe_idx) >= self.keyframe_interval:
            self.keyframe_depth = new_depth.clone()
            self.keyframe_idx   = self.frame_count

    def _log_buffer(self, tensor: torch.Tensor, name: str):
        arr = tensor.cpu().numpy()
        arr = np.squeeze(arr)   # drop any dims of size 1
        if arr.ndim != 2:
            # collapse leading axes if necessary
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        vis = ((arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        self.gui.addBuffer(name, bgr)
