#!/usr/bin/env python3
# depth_generator.py – reusable Distill-Any-Depth wrapper with 3 compile tiers
# -----------------------------------------------------------------------------
# tiers:
#   • "fast"  – shortest startup (≈1 s)      torch.compile mode="reduce-overhead"
#   • "mid"   – balanced (≈2 – 2.5 s)        mode="default"
#   • "best"  – full autotune (3 – 5 s)       mode="max-autotune"
#
# usage:
#   gen = DepthGenerator(compile_level="mid")
#   depth_bytes = gen.process(rgba_np, W, H)
# -----------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthGenerator:
    """
    Parameters
    ----------
    model_id      : str   HF model id (default small distilled DepthAnything)
    compile_level : str   "fast" | "mid" | "best"

    process(rgba_np, w, h) -> bytes
        rgba_np : H×W×4  uint8  (RGB + A, row-major)
        w, h    : desired output size (usually original width, height)
    """

    _LEVEL_TO_MODE = {
        "fast": "reduce-overhead",
        "mid":  "default",
        "best": "max-autotune",
    }

    def __init__(
        self,
        model_id: str = "xingyang1/Distill-Any-Depth-Small-hf",
        compile_level: str = "mid"
    ):
        compile_level = compile_level.lower()
        if compile_level not in self._LEVEL_TO_MODE:
            raise ValueError("compile_level must be 'fast', 'mid', or 'best'")

        # -------- Inductor cache & CuDNN autotune ---------------------------
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            os.path.expanduser("~/.cache/torch_inductor")
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        # -------- load & compile ---------------------------------------------
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        eager = (
            AutoModelForDepthEstimation
            .from_pretrained(model_id)
            .to("cuda")
            .eval()
        )

        mode = self._LEVEL_TO_MODE[compile_level]
        print(f"torch.compile mode='{mode}'  (level={compile_level})")
        self.model = torch.compile(
            eager,
            fullgraph=True,
            dynamic=False,
            mode=mode
        )

        # capture CUDA graph once
        example = torch.randn(1, 3, 384, 384, device="cuda")
        with torch.inference_mode():
            _ = self.model(example)
        torch.cuda.synchronize()

        # -------- constants -------------------------------------------------
        self.RESIZE_HW = (
            self.processor.size["height"],
            self.processor.size["width"]
        )
        self.MEAN = torch.tensor(
            self.processor.image_mean,
            dtype=torch.float32,
            device="cuda"
        ).view(1, 3, 1, 1)
        self.STD = torch.tensor(
            self.processor.image_std,
            dtype=torch.float32,
            device="cuda"
        ).view(1, 3, 1, 1)

        self.stream = torch.cuda.Stream()
        self._pinned: dict[tuple[int, int], torch.Tensor] = {}

    def process(self, rgba: np.ndarray, w: int, h: int) -> bytes:
        """
        Returns
        -------
        bytes_buffer : bytes
            Depth map (float32 little-endian, size h*w*4)
        """
        # ---- host→device and preprocess ----------------------------------
        with torch.cuda.stream(self.stream):
            img = (
                torch.from_numpy(rgba)
                .permute(2, 0, 1)[:3]
                .unsqueeze(0)
                .pin_memory()
                .to("cuda", non_blocking=True, dtype=torch.float32) / 255
            )
            img = (img[:, [2, 1, 0], ...] - self.MEAN) / self.STD
            img = F.interpolate(
                img,
                size=self.RESIZE_HW,
                mode="bilinear",
                align_corners=False
            )
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()

        # ---- inference ----------------------------------------------------
        with torch.inference_mode():
            out = self.model(pixel_values=img).predicted_depth
            if out.ndim == 3:
                out = out.unsqueeze(1)
        torch.cuda.synchronize()

        # ---- upsample + copy to pinned host ------------------------------
        depth = F.interpolate(
            out,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )[0, 0]
        host = self._pinned.setdefault(
            (h, w),
            torch.empty((h, w), dtype=torch.float32, pin_memory=True)
        )
        host.copy_(depth, non_blocking=True)
        torch.cuda.synchronize()

        # ---- return depth buffer -----------------------------------------
        return host.numpy().tobytes()
