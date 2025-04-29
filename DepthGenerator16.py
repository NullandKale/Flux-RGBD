#!/usr/bin/env python3
# DepthGenerator16.py – fused CUDA‐compiled pipeline + AMP + dynamic graph recapture with GUI integration
# -----------------------------------------------------------------------------
# Tiers:
#   • "fast"  – shortest startup (≈1 s)      torch.compile mode="reduce-overhead"
#   • "mid"   – balanced (≈2–2.5 s)          mode="default"
#   • "best"  – full autotune (3–5 s)        mode="max-autotune"
#
# Alignment fix + AMP: use align_corners=True to preserve pixel‐to‐pixel mapping
# and optional torch.amp.autocast for half‐precision inference.

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.amp
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Optional GUI interface
try:
    from GUI import GUI
except ImportError:
    GUI = None

class DepthGenerator:
    _LEVEL_TO_MODE = {
        "fast": "reduce-overhead",
        "mid":  "default",
        "best": "max-autotune",
    }

    def __init__(
        self,
        model_id: str = "xingyang1/Distill-Any-Depth-Small-hf",
        compile_level: str = "mid",
        use_amp: bool = True,
        gui: GUI = None,
    ):
        compile_level = compile_level.lower()
        if compile_level not in self._LEVEL_TO_MODE:
            raise ValueError("compile_level must be 'fast', 'mid', or 'best'")

        # --- CUDA tuning ---
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            os.path.expanduser("~/.cache/torch_inductor"),
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark       = True
        torch.set_float32_matmul_precision("high")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        self.dev     = torch.device("cuda:0")
        self.use_amp = use_amp
        self.dtype   = torch.float16 if self.use_amp else torch.float32
        self.gui     = gui

        # --- load & compile model ---
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        eager = (
            AutoModelForDepthEstimation
            .from_pretrained(model_id)
            .to(self.dev)
            .eval()
        )
        if self.use_amp:
            eager = eager.half()

        mode = self._LEVEL_TO_MODE[compile_level]
        self.model = torch.compile(
            eager,
            fullgraph=True,
            dynamic=False,
            mode=mode,
        )

        # warm‐up for CUDA graphs
        example = torch.randn(1, 3, 384, 384, device=self.dev, dtype=self.dtype)
        with torch.inference_mode():
            _ = self.model(pixel_values=example)
        torch.cuda.synchronize()

        # --- normalization constants ---
        h0, w0 = (
            self.processor.size["height"],
            self.processor.size["width"],
        )
        self.RESIZE_HW = (h0, w0)
        self.MEAN = torch.tensor(
            self.processor.image_mean,
            dtype=self.dtype,
            device=self.dev
        ).view(1, 3, 1, 1)
        self.STD = torch.tensor(
            self.processor.image_std,
            dtype=self.dtype,
            device=self.dev
        ).view(1, 3, 1, 1)

        # --- host buffer cache ---
        self._pinned_out: dict[tuple[int,int], torch.Tensor] = {}

    def process(self, rgba: np.ndarray, w: int, h: int) -> bytes:
        """
        Returns:
            Depth map as float32 little‐endian bytes (length h*w*4).
        Also logs inference time to GUI if provided.
        """
        start = time.perf_counter()

        # upload to pinned host memory
        host_in = torch.from_numpy(rgba).pin_memory()
        # transfer to GPU & cast
        img = (
            host_in
            .to(self.dev, non_blocking=True)
            .permute(2, 0, 1)[:3]
            .unsqueeze(0)
            .to(self.dtype)
            / 255.0
        )

        # normalize
        img = (img - self.MEAN) / self.STD

        # downsample to model square
        img = F.interpolate(
            img,
            size=self.RESIZE_HW,
            mode="bilinear",
            align_corners=True
        )

        # inference under autocast if AMP enabled
        with torch.inference_mode(), \
             torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
            pred = self.model(pixel_values=img).predicted_depth
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)

        # upsample back to original resolution
        depth = F.interpolate(
            pred,
            size=(h, w),
            mode="bilinear",
            align_corners=True
        )[0, 0]

        # copy into pinned float32 host buffer
        host_out = self._pinned_out.setdefault(
            (h, w),
            torch.empty((h, w), dtype=torch.float32, pin_memory=True)
        )
        host_out.copy_(depth, non_blocking=True)
        torch.cuda.synchronize()

        # calculate elapsed
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return host_out.cpu().numpy().tobytes()
