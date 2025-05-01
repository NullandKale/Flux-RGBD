from collections import deque
import numpy as np
from typing import Deque, Optional, Tuple, List

class Frame:
    """
    Holds:
      - raw RGB image
      - native depth maps (raw and filtered variants)
      - various per-frame buffers produced by DepthFilter
      - visualization images and numeric metrics
    """

    def __init__(self, rgb: np.ndarray):
        # -- raw inputs --
        self.rgb: np.ndarray = rgb                                 # H×W×3 uint8 BGR
        self.depth_bytes: Optional[bytes] = None                   # raw bytes from depth model
        self.d_nat: Optional[np.ndarray] = None                    # H×W float32 native depth

        # -- processed depths --
        self.d_filtered: Optional[np.ndarray] = None               # H×W float32 temporally filtered depth
        self.d_autofocus: Optional[np.ndarray] = None              # H×W float32 autofocus-adjusted depth
        self.d_text: Optional[np.ndarray] = None                   # H×W float32 text-masked depth

        # -- buffers added by DepthFilter --
        self.raw_depth: Optional[np.ndarray] = None    # H×W float32 copy of original depth passed to filter
        self.edge_map: Optional[np.ndarray] = None     # H×W float32 Sobel gradient magnitude
        self.variance_map: Optional[np.ndarray] = None # H×W float32 per-pixel variance across history
        self.var_weight: Optional[np.ndarray] = None   # H×W float32 variance-based blending weight
        self.gradient_map: Optional[np.ndarray] = None# H×W float32 per-pixel temporal gradient
        self.grad_weight: Optional[np.ndarray] = None # H×W float32 gradient-based blending weight
        self.motion_map: Optional[np.ndarray] = None  # H×W float32 combined motion mask
        self.history_avg: Optional[np.ndarray] = None # H×W float32 weighted average of history frames
        self.diff_ratio: Optional[np.ndarray] = None  # H×W float32 normalized difference ratio aligned vs reference
        self.blend_weight: Optional[np.ndarray] = None# H×W float32 final blend weight used for fusion
        self.diff_output: Optional[np.ndarray] = None # H×W float32 absolute difference filtered vs raw

        # -- face detection --
        self.boxes: List[Tuple[int, int, int, int]] = []
        self.scores: List[float] = []

        # -- visualizations (all BGR uint8 H×W×3) --
        self.vis: Optional[np.ndarray] = None            # side-by-side RGB|depth visualization
        self.depth_vis_img: Optional[np.ndarray] = None # H×W×3 uint8 grayscale depth view
        self.raw_depth_vis: Optional[np.ndarray] = None # H×W×3 uint8 view of raw_depth
        self.edge_map_vis: Optional[np.ndarray] = None  # H×W×3 uint8 view of edge_map
        self.variance_map_vis: Optional[np.ndarray] = None # H×W×3 uint8 view of variance_map
        self.gradient_map_vis: Optional[np.ndarray] = None # H×W×3 uint8 view of gradient_map
        self.motion_map_vis: Optional[np.ndarray] = None  # H×W×3 uint8 view of motion_map
        self.history_avg_vis: Optional[np.ndarray] = None # H×W×3 uint8 view of history_avg
        self.diff_ratio_vis: Optional[np.ndarray] = None  # H×W×3 uint8 view of diff_ratio
        self.diff_output_vis: Optional[np.ndarray] = None # H×W×3 uint8 view of diff_output
        self.text_mask_vis: Optional[np.ndarray] = None   # H×W×3 uint8 view of OCR text mask
        self.text_confidence_vis: Optional[np.ndarray] = None # H×W×3 uint8 confidence visualization

        # -- numeric metrics --
        self.ms_depth: Optional[float] = None
        self.ms_filter: Optional[float] = None
        self.ms_autofocus: Optional[float] = None
        self.ms_text: Optional[float] = None
        self.ms_write: Optional[float] = None
        self.ms_frame: Optional[float] = None
        self.fps: Optional[float] = None
        self.eta_s: Optional[float] = None
        self.face_count: Optional[int] = None
        self.scene_cut: Optional[float] = None       # 1.0 if scene cut detected, else 0.0
        self.rgb_change: Optional[float] = None      # mean absolute RGB change from previous frame

    def depth(self, depth_model, width: int, height: int) -> np.ndarray:
        if self.d_nat is None:
            self.depth_bytes = depth_model.process(self.rgb, width, height)
            self.d_nat = np.frombuffer(self.depth_bytes, np.float32).reshape(height, width)
        return self.d_nat

    def compute_depth_vis(self):
        """
        Fill self.depth_vis_img with a 3-channel grayscale view of depth.
        """
        d = self.d_text if self.d_text is not None else self.d_filtered
        mn, mx = float(d.min()), float(d.max())
        if mx - mn < 1e-6:
            mx = mn + 1e-6
        g = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
        self.depth_vis_img = np.repeat(g[..., None], 3, axis=2)
        return self.depth_vis_img

    def compute_side_by_side(self):
        """Concatenate RGB and depth_vis_img."""
        if self.depth_vis_img is None:
            self.compute_depth_vis()
        self.vis = np.concatenate((self.rgb, self.depth_vis_img), axis=1)
        return self.vis

class FrameHistory:
    """Keeps the last N Frame objects for temporal passes (e.g. filtering)."""
    def __init__(self, maxlen: int):
        self._queue: Deque[Frame] = deque(maxlen=maxlen)

    def append(self, frame: Frame):
        self._queue.append(frame)

    def latest(self, n: int) -> List[Frame]:
        return list(self._queue)[-n:]
