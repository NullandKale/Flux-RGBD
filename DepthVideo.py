#!/usr/bin/env python3
# DepthVideo.py – RGB | Depth side-by-side video with tqdm progress bar
# --------------------------------------------------------------------
# needs: opencv-python, numpy, torch, transformers, tqdm, ffmpeg (incl. nvenc)
#
#   python DepthVideo.py input.mp4                 # NVENC, 512-px depth input
#   python DepthVideo.py input.mp4 --cpu           # libx264 instead of NVENC
#   python DepthVideo.py input.mp4 --max_res 720
#   python DepthVideo.py /path/to/folder            # process all videos in folder (smallest first)
# --------------------------------------------------------------------

import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

from VideoUtils import (
    VideoFrameGenerator,
    get_video_frame_count,
    FFMPEGVideoWriter,
)
from DepthGenerator import DepthGenerator  # global depth engine

# --------------------------------------------------------------------
def depth_to_gray(depth32: np.ndarray) -> np.ndarray:
    """Normalize depth → 3-channel uint8 grayscale."""
    mn, mx = float(depth32.min()), float(depth32.max())
    if mx - mn < 1e-6:
        mx = mn + 1e-6
    g = ((depth32 - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)  # H×W×3

# --------------------------------------------------------------------
def process_video(path_in: str, max_res: int, use_nvenc: bool):
    base = os.path.splitext(os.path.basename(path_in))[0]
    outdir = "./output"
    os.makedirs(outdir, exist_ok=True)
    path_out = os.path.join(outdir, f"{base}_rgbd.mp4")

    print(f"\nProcessing '{path_in}' → '{path_out}'")
    frames = VideoFrameGenerator(path_in, process_length=-1, max_res=max_res)
    depth = DepthGenerator(model_id="xingyang1/Distill-Any-Depth-Large-hf", compile_level="mid")

    writer = FFMPEGVideoWriter(
        output_file=path_out,
        fps=frames.fps,
        width=frames.width * 2,  # double width for side-by-side
        height=frames.height,
        audio_file=path_in,
        debug=False,
        use_nvenc=use_nvenc
    )

    total_frames = get_video_frame_count(path_in) or int(frames.frame_count)

    with tqdm(total=total_frames, unit="frame", desc=base) as bar:
        for _, rgb in frames:
            # depth inference -------------------------------------------------
            d_bytes, _ = depth.process(rgb, frames.width, frames.height)
            d32 = np.frombuffer(d_bytes, np.float32).reshape(frames.height, frames.width)
            d_vis = depth_to_gray(d32)  # H×W×3 uint8

            side = np.concatenate((rgb, d_vis), axis=1)  # H × (2W) × 3
            writer.write_frame(side)
            bar.update(1)

    writer.release()
    print(f"Saved → {path_out}")

# --------------------------------------------------------------------
def is_video_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.mpg', '.mpeg'}

# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB + Depth video generator")
    parser.add_argument("input", help="input video file or folder")
    parser.add_argument("--max_res", type=int, default=1280,
                        help="max resolution fed into depth net (default 512)")
    parser.add_argument("--cpu", action="store_true",
                        help="use libx264 (CPU) instead of NVENC HEVC")
    args = parser.parse_args()

    input_path = args.input
    use_nvenc = not args.cpu

    if os.path.isdir(input_path):
        # collect video files in folder, sort by size ascending
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f)) and is_video_file(f)
        ]
        if not files:
            print(f"No video files found in folder '{input_path}'.")
            sys.exit(1)
        files.sort(key=lambda p: os.path.getsize(p))
        for vid in files:
            process_video(vid, max_res=args.max_res, use_nvenc=use_nvenc)
    else:
        if not os.path.isfile(input_path):
            print(f"Input '{input_path}' does not exist.")
            sys.exit(1)
        if not is_video_file(input_path):
            print(f"Input '{input_path}' is not a supported video file.")
            sys.exit(1)
        process_video(input_path, max_res=args.max_res, use_nvenc=use_nvenc)
