#!/usr/bin/env python3
# TextDepthFilter.py – GPU-accelerated OCR mask refinement
# ----------------------------------------------------------------------------
import easyocr
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from Frame import Frame

class TextDepthFilter:
    """
    Filters a depth map by:
      1) CPU: detecting text boxes via EasyOCR
      2) GPU: computing Sobel edges, median-blur uniformity, and fusing them
      3) CPU: rasterizing each box’s polygon mask and filling depth
    """

    def __init__(
        self,
        gui: Optional["GUI"] = None,
        lang_list: List[str] = ["en"],
        gpu: bool = True,
        pad: int = 2,

        # thresholds
        edge_threshold: float      = 0.2,
        color_threshold: float     = 0.2,
        combined_threshold: float  = 0.2,

        median_ksize: int          = 13,
    ):
        self.gui               = gui
        self.reader            = easyocr.Reader(lang_list, gpu=gpu)
        self.pad               = pad
        self.edge_threshold    = edge_threshold
        self.color_threshold   = color_threshold
        self.combined_threshold= combined_threshold
        self.median_ksize      = median_ksize

        # setup device
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

        # prepare Sobel kernels on device
        sx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=self.device)
        sy = sx.t()
        self.sobel_x = sx.view(1,1,3,3)
        self.sobel_y = sy.view(1,1,3,3)

    def filter_frame(self, frame: Frame) -> np.ndarray:
        # --- 0) pick depth array ---
        if frame.d_filtered is not None:
            depth_full = frame.d_filtered
        elif frame.d_autofocus is not None:
            depth_full = frame.d_autofocus
        else:
            depth_full = frame.d_nat
        H, W = depth_full.shape

        # --- 1) detect text boxes on CPU ---
        boxes: List[Tuple[int,int,int,int]] = []
        gray_cpu = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY)
        for bbox, _, _ in self.reader.readtext(gray_cpu, detail=1):
            x0 = max(0, int(min(pt[0] for pt in bbox)))
            y0 = max(0, int(min(pt[1] for pt in bbox)))
            x1 = min(W, int(max(pt[0] for pt in bbox)))
            y1 = min(H, int(max(pt[1] for pt in bbox)))
            boxes.append((x0,y0,x1,y1))

        # --- 2) move gray to GPU and compute Sobel edges ---
        # normalize to [0,1]
        gray_t = torch.from_numpy(gray_cpu).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.0
        Gx = F.conv2d(gray_t, self.sobel_x, padding=1)
        Gy = F.conv2d(gray_t, self.sobel_y, padding=1)
        grad = torch.sqrt(Gx*Gx + Gy*Gy)
        grad = grad / (grad.amax()+1e-6)  # normalize

        # --- 3) GPU median‐blur color‐uniformity via unfold + median ---
        k = self.median_ksize
        pad = k//2
        # unfold to get local patches
        patches = F.unfold(gray_t, kernel_size=k, padding=pad)  # [1, k*k, H*W]
        median_vals,_ = patches.median(dim=1, keepdim=True)     # [1,1,H*W]
        median_img = median_vals.view(1,1,H,W)
        uniform = torch.abs(gray_t - median_img)                # difference
        uniform = uniform.clamp(0,1)                            # already in [0,1]

        # --- 4) fuse cues on GPU ---
        fused = 0.5*grad + 0.5*uniform
        fused = fused.clamp(0,1)  # [1,1,H,W]

        # bring fused maps back to CPU
        grad_cpu    = grad.squeeze().cpu().numpy()
        uniform_cpu = uniform.squeeze().cpu().numpy()
        fused_cpu   = fused.squeeze().cpu().numpy()

        # --- 5) build final mask by refining inside each box ---
        mask = np.zeros((H,W), dtype=bool)
        for (x0,y0,x1,y1) in boxes:
            xb0, yb0 = max(0,x0-self.pad), max(0,y0-self.pad)
            xb1, yb1 = min(W,x1+self.pad), min(H,y1+self.pad)

            roi_g = grad_cpu[yb0:yb1, xb0:xb1]
            roi_u = uniform_cpu[yb0:yb1, xb0:xb1]
            roi_f = fused_cpu[yb0:yb1, xb0:xb1]

            # within-box stroke mask
            roi_mask = (roi_g > self.edge_threshold) & \
                       (roi_u > self.color_threshold) & \
                       (roi_f > self.combined_threshold)

            mask[yb0:yb1, xb0:xb1] = roi_mask

        # --- 6) fill text strokes in depth via median‐of‐border per box ---
        new_depth = depth_full.copy()
        for (x0,y0,x1,y1) in boxes:
            xb0, yb0 = max(0,x0-self.pad), max(0,y0-self.pad)
            xb1, yb1 = min(W,x1+self.pad), min(H,y1+self.pad)
            if xb1<=xb0 or yb1<=yb0: continue

            border = np.concatenate([
                new_depth[yb0:y0, xb0:xb1].ravel(),
                new_depth[y1:yb1, xb0:xb1].ravel(),
                new_depth[yb0:yb1, xb0:x0].ravel(),
                new_depth[yb0:yb1, x1:xb1].ravel(),
            ]) if True else np.array([], dtype=new_depth.dtype)

            if border.size:
                fill = np.median(border)
                roi  = new_depth[y0:y1, x0:x1]
                roi[mask[y0:y1, x0:x1]] = fill
                new_depth[y0:y1, x0:x1] = roi

        frame.d_text = new_depth

        # --- 7) GPU mask visualization (back on CPU) ---
        vis = cv2.cvtColor((fused_cpu*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        vis[mask] = (255,0,255)
        frame.text_confidence_vis = vis
        if self.gui:
            self.gui.addBuffer("text_confidence", vis)

        return new_depth
