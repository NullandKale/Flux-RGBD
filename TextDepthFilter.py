import easyocr  # pip install easyocr
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional

class TextDepthFilter:
    """
    TextDepthFilter

    Goal:
        Detect text regions in an RGB image at half resolution, compute a per-pixel
        “text-ness” confidence map there, upscale the binary mask to full resolution,
        and fill those regions in the full-resolution depth map to remove text artifacts.

    Key Steps:
        1. Downsample RGB to ½ size → rgb_half.
        2. OCR on rgb_half → bounding boxes in half-res coords.
        3. For each box in half-res:
           a. Compute Sobel edge magnitude.
           b. Compute color-distance from patch median.
           c. Fuse into soft confidence ∈ [0,1].
           d. Threshold to a binary mask_ds.
        4. Upsample mask_ds to full resolution (nearest).
        5. Visualize confidence in GUI.
        6. Use mask_full to fill text regions in full-res depth via median of border pixels.
    """

    def __init__(
        self,
        gui: Optional["GUI"] = None,
        lang_list: List[str] = ["en"],
        gpu: bool = True,
        edge_thresh: float = 0.2,
        color_thresh: float = 0.1,
        pad: int = 2,
    ):
        """
        Initialize OCR reader and edge/color thresholds.

        Parameters:
            gui          Optional GUI for visualization.
            lang_list    Languages for EasyOCR.
            gpu          If True, EasyOCR uses CUDA.
            edge_thresh  Gradient threshold for edge confidence.
            color_thresh Color-distance threshold for color confidence.
            pad          Border width (in full-res pixels) for depth fill.
        """
        self.gui = gui
        self.edge_thresh = edge_thresh
        self.color_thresh = color_thresh
        self.pad = pad

        # Initialize EasyOCR reader
        try:
            self.reader = easyocr.Reader(lang_list, gpu=gpu)
        except Exception as e:
            print(f"[TextDepthFilter] EasyOCR init failed: {e}")
            self.reader = None

        # Prepare Sobel kernels (CPU); will move to device later
        sx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        sy = sx.t()
        self.sobel_x = sx.view(1,1,3,3)
        self.sobel_y = sy.view(1,1,3,3)
        self._kernels_registered = False

    def detect_text(self, image: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """
        Run OCR on the given RGB image (half-res) to get text bounding boxes.

        Returns:
            List of (x0,y0,x1,y1) in image pixel coords.
        """
        if self.reader is None:
            return []
        results = self.reader.readtext(image)
        boxes: List[Tuple[int,int,int,int]] = []
        for box, _, _ in results:
            xs = [int(pt[0]) for pt in box]
            ys = [int(pt[1]) for pt in box]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes

    def filter(
        self,
        d32: np.ndarray,
        rgb: np.ndarray
    ) -> np.ndarray:
        """
        Filter the input full-res depth by removing text artifacts.

        Parameters:
            d32  Full-resolution depth map [H,W], float32.
            rgb  Full-resolution RGB image [H,W,3], uint8 or float.

        Returns:
            new_depth: np.ndarray [H,W], float32 — depth with text regions filled.
        """
        # Select compute device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Full-res depth tensor (we never downsample depth)
        depth_full = torch.from_numpy(d32).to(device)  # [H,W]
        H, W = depth_full.shape

        # Create half-res RGB for OCR and mask generation
        H2, W2 = H // 2, W // 2
        rgb_half = cv2.resize(rgb, (W2, H2), interpolation=cv2.INTER_LINEAR)

        # OCR on half-res image
        boxes = self.detect_text(rgb_half)

        # Convert rgb_half to torch tensor [3,H2,W2]
        img_ds = torch.from_numpy(rgb_half).float().permute(2,0,1).to(device) / 255.0

        # Prepare half-res mask & confidence
        mask_ds = torch.zeros((H2, W2), dtype=torch.bool, device=device)
        conf_ds = torch.zeros((H2, W2), dtype=torch.float32, device=device)

        # Move Sobel kernels to device once
        if not self._kernels_registered:
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)
            self._kernels_registered = True

        # Process each bounding box at half resolution
        for x0, y0, x1, y1 in boxes:
            # Clamp box to half-res image
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(W2, x1), min(H2, y1)
            if x1c <= x0c or y1c <= y0c:
                continue

            patch = img_ds[:, y0c:y1c, x0c:x1c]  # [3,ph,pw]
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < 3 or pw < 3:
                continue

            # 1) Sobel edge magnitude
            gray = (0.2989*patch[0] + 0.5870*patch[1] + 0.1140*patch[2]).unsqueeze(0).unsqueeze(0)
            Gx = F.conv2d(gray, self.sobel_x, padding=1)
            Gy = F.conv2d(gray, self.sobel_y, padding=1)
            grad = torch.sqrt(Gx*Gx + Gy*Gy).squeeze()
            edge_conf = grad / (grad.max() + 1e-6)

            # 2) Color-distance confidence
            flat = patch.reshape(3, -1)
            med = flat.median(dim=1).values[:,None,None]
            dist = torch.norm(patch - med, dim=0)
            color_conf = (self.color_thresh - dist).clamp(0, self.color_thresh) / self.color_thresh

            # 3) Fuse confidences
            pix_conf = 0.5 * edge_conf + 0.5 * color_conf

            # 4) Horizontal weighting (centered)
            coords = torch.linspace(0, pw-1, pw, device=device)
            hw = 1 - (coords - (pw-1)/2).abs() / ((pw-1)/2 + 1e-6)
            pix_conf *= hw.unsqueeze(0).expand(ph, pw)

            # 5) Area weighting (small boxes downweighted)
            pix_conf *= (ph * pw) / (H2 * W2)

            # Store soft confidence and binary mask
            conf_ds[y0c:y1c, x0c:x1c] = pix_conf
            mask_ds[y0c:y1c, x0c:x1c] = pix_conf > 0.0

        # Upsample mask & confidence to full resolution
        mask_full = F.interpolate(
            mask_ds.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode='nearest'
        ).squeeze().bool()
        conf_full = F.interpolate(
            conf_ds.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze().clamp(0, 1)

        # Visualize confidence in GUI:
        # - background: greyscale = confidence  
        # - overlay: magenta (R+B) where mask is True
        if self.gui:
            # start with greyscale background
            grey = conf_full  # [H,W] ∈ [0,1]
            vis = torch.stack([grey, grey, grey], dim=0)  # [3,H,W]

            # overlay magenta on mask pixels
            mask = mask_full  # boolean [H,W]
            vis[0, mask] = 1.0   # red channel = 1
            vis[1, mask] = 0.0   # green channel = 0
            vis[2, mask] = 1.0   # blue channel = 1

            # convert to uint8 RGB
            vis = (vis * 255).byte().cpu().numpy().transpose(1,2,0)
            self.gui.addBuffer("text_confidence", vis)

        # Fill depth in masked regions using median of full-res border pixels
        new_depth = depth_full.clone()
        pad = self.pad
        ys, xs = torch.where(mask_full)
        if ys.numel():
            for x0h, y0h, x1h, y1h in boxes:
                # scale box from half-res to full-res coords
                x0f, y0f = x0h * 2, y0h * 2
                x1f, y1f = min(W, x1h * 2), min(H, y1h * 2)
                if x1f <= x0f or y1f <= y0f:
                    continue
                xb0, yb0 = max(0, x0f - pad), max(0, y0f - pad)
                xb1, yb1 = min(W, x1f + pad), min(H, y1f + pad)

                # sample border pixels
                top    = new_depth[yb0:yb0+pad, xb0:xb1].reshape(-1)
                bottom = new_depth[yb1-pad:yb1, xb0:xb1].reshape(-1)
                left   = new_depth[yb0:yb1, xb0:xb0+pad].reshape(-1)
                right  = new_depth[yb0:yb1, xb1-pad:xb1].reshape(-1)
                border = torch.cat([top, bottom, left, right], dim=0)

                fill_val = border.median() if border.numel() else torch.tensor(0.0, device=device)
                new_depth[y0f:y1f, x0f:x1f] = fill_val

        return new_depth.cpu().numpy()
