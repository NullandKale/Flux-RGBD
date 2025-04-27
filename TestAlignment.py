#!/usr/bin/env python3
"""
TestAlignment.py – runs DepthVideo.py in mode -1 or -2 and computes an alignment score [0..1]

Usage:
    python TestAlignment.py <input_video> [--delay] [--seed SEED] [--diff-images] [--no-diff] [--threshold N]

By default, runs same-frame side-by-side (mode -1). With --delay, uses delayed-right (mode -2).
Generates a full diff video (_diff.mp4) or per-frame diff images.
Instead of raw diffs, computes a normalized mean-diff score in [0..1] (0=perfect match, 1=invert).
When doing full-diff, scores ALL frames; with --diff-images, scores only selected frames.
Exits 0 if alignment score ≤ threshold, else 1.
"""

import argparse
import subprocess
import sys
import os
import random
import cv2
import numpy as np


def run_depth_video(input_path: str, useDelayed: bool = False) -> str:
    script = os.path.join(os.path.dirname(__file__), 'DepthVideo.py')
    mode = '-2' if useDelayed else '-1'
    cmd = [sys.executable, script, '-q', '--mode', mode, input_path]
    print("Running command:", ' '.join(cmd))
    subprocess.run(cmd, check=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join('output', f"{base}_rgbd.mp4")


def select_frame_indices(total_frames: int, seed: int = None) -> list[int]:
    if total_frames < 2:
        return [0]
    first, last = 0, total_frames - 1
    middle = list(range(1, last))
    rnd = random.Random(seed)
    sample = rnd.sample(middle, min(5, len(middle))) if middle else []
    return [first, last] + sample


def compute_diff_stats(frame: np.ndarray) -> tuple[int, float]:
    h, w = frame.shape[:2]
    half = w // 2
    left, right = frame[:, :half], frame[:, half:]
    diff = cv2.absdiff(left, right)
    return int(diff.max()), float(diff.mean())


def write_full_diff_video(side_vid: str, out_diff: str) -> tuple[float, float]:
    cap = cv2.VideoCapture(side_vid)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {side_vid}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//2)
    h2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_diff, fourcc, fps, (w2, h2))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mean_diffs = []
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        _, mean_d = compute_diff_stats(frame)
        mean_diffs.append(mean_d)
        gray = cv2.cvtColor(cv2.absdiff(frame[:, :w2], frame[:, w2:]), cv2.COLOR_BGR2GRAY)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cap.release(); writer.release()
    overall_mean = float(np.mean(mean_diffs)) if mean_diffs else 0.0
    normalized = overall_mean / 255.0
    return normalized, float(total)


def write_diff_images(side_vid: str, indices: list[int], out_dir: str) -> dict[int, float]:
    cap = cv2.VideoCapture(side_vid)
    os.makedirs(out_dir, exist_ok=True)
    norm_stats = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            norm_stats[idx] = None
            continue
        _, mean_d = compute_diff_stats(frame)
        norm_stats[idx] = mean_d / 255.0
        h, w = frame.shape[:2]
        half = w // 2
        diff = cv2.absdiff(frame[:, :half], frame[:, half:])
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:04d}_diff.jpg"), img)
    cap.release()
    return norm_stats


def main():
    p = argparse.ArgumentParser(description="Test alignment via normalized diff score")
    p.add_argument('input', help='Input video')
    p.add_argument('--delay', action='store_true', help='Use delayed mode (-2)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--no-diff', action='store_true')
    p.add_argument('--diff-images', action='store_true')
    p.add_argument('--threshold', type=float, default=0.01,
                   help='Max allowed normalized mean-diff (0..1), default 0.005')
    args = p.parse_args()

    mode_desc = 'delayed' if args.delay else 'same-frame'
    print(f"Running DepthVideo.py in {mode_desc} mode on '{args.input}'...")
    side_vid = run_depth_video(args.input, useDelayed=args.delay)
    print(f"Side-by-side: {side_vid}\n")

    cap = cv2.VideoCapture(side_vid)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        print("No frames."); sys.exit(1)

    indices = select_frame_indices(total, seed=args.seed)
    indices.sort()
    print(f"Selected indices: {indices}\n")

    if not args.no_diff:
        base = os.path.splitext(os.path.basename(side_vid))[0]
        if args.diff_images:
            out_dir = os.path.join('output', f"{base}_diff_images")
            print(f"Generating diff images at {out_dir}\n")
            stats = write_diff_images(side_vid, indices, out_dir)
            overall = np.mean([v for v in stats.values() if v is not None])
        else:
            diff_vid = os.path.join('output', f"{base}_diff.mp4")
            print(f"Writing full diff video to {diff_vid}\n")
            overall, _ = write_full_diff_video(side_vid, diff_vid)
    else:
        stats = {idx: None for idx in indices}
        overall = 0.0

    print(f"Normalized mean-diff score: {overall:.6f} (threshold {args.threshold})")
    if args.diff_images:
        print("Frame | Normalized Diff")
        print("------|----------------")
        for idx in indices:
            v = stats.get(idx)
            mark = '' if v is not None and v <= args.threshold else '  *'
            print(f"{idx:5d} | {v if v is not None else 'N/A':>12}{mark}")

    if overall <= args.threshold:
        print("\nAlignment PASS.")
        sys.exit(0)
    else:
        print("\nAlignment FAIL.")
        sys.exit(1)

if __name__ == '__main__':
    main()
