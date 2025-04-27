#!/usr/bin/env python3
# DepthVideo.py – RGB | Depth side-by-side writer with GUI abstraction
# -------------------------------------------------------------------------------
import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

from VideoUtils         import VideoFrameGenerator, get_video_frame_count, FFMPEGVideoWriter
from DepthGenerator16   import DepthGenerator
from DepthFilter        import DepthFilter
from DepthAutoFocus     import DepthAutoFocus
from DetectFace         import FaceDetector
from GUI                import GUI


def round_up32(x: int) -> int:
    return ((x + 31) // 32) * 32


def depth_to_gray(d: np.ndarray) -> np.ndarray:
    mn, mx = float(d.min()), float(d.max())
    if mx - mn < 1e-6:
        mx = mn + 1e-6
    g = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='RGB + Depth video generator')
    p.add_argument('input',            help='input file or folder')
    p.add_argument('--cpu',            action='store_true', help='use libx264 instead of NVENC')
    p.add_argument('--no_preview',     action='store_true', help='disable preview window')
    p.add_argument('--skip_ui','-q',   action='store_true', help='alias for --no_preview')
    p.add_argument('--mode', type=int, choices=[-2,-1,0,1,2,3], default=3,
                   help='-2=delay-right, -1=same-frame, 0=depth,1=+filter,2=+faces,3=+autofocus')
    p.add_argument('--max_res', type=int, default=720,
                   help='maximum frame height (preserves aspect ratio)')
    args = p.parse_args()

    use_nvenc = not args.cpu
    show_prev = not (args.no_preview or args.skip_ui)

    # initialize GUI once
    gui = GUI(window_name='Depth Preview',
              window_size=(2560, 720),
              fps=30) if show_prev else None

    # depth model can emit timing stats via gui
    depth_model = DepthGenerator(
        model_id='xingyang1/Distill-Any-Depth-Small-hf',
        compile_level='fast',
        gui=gui
    )

    # face detector will log detections via gui
    face_detector = FaceDetector(
        device='cuda',
        gui=gui
    ) if args.mode >= 2 else None

    def process_video(path_in):
        do_filter    = (args.mode >= 1)
        do_faces     = (args.mode >= 2)
        do_autofocus = (args.mode >= 3)

        frames = VideoFrameGenerator(path_in,
                                     process_length=-1,
                                     max_res=args.max_res)
        it = iter(frames)
        try:
            _, prev_rgb = next(it)
        except StopIteration:
            return

        H, W = prev_rgb.shape[:2]
        base   = os.path.splitext(os.path.basename(path_in))[0]
        outdir = './output'; os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, f'{base}_rgbd.mp4')
        print(f"\nProcessing '{path_in}' → '{out_fp}'")

        # filters / autofocus also get gui hooks
        depth_filter = DepthFilter(
            device='cuda',
            fps=frames.fps,
            gui=gui
        ) if do_filter else None

        autofocus = DepthAutoFocus(
            fps=frames.fps,
            target=0.5,
            gui=gui
        ) if do_autofocus else None

        writer = FFMPEGVideoWriter(
            output_file=out_fp,
            fps=frames.fps,
            width=W*2,
            height=H,
            audio_file=path_in,
            debug=False,
            use_nvenc=use_nvenc
        )

        total = get_video_frame_count(path_in) or int(frames.frame_count)
        prev_tick = cv2.getTickCount()

        def run_frame(curr_rgb):
            nonlocal prev_rgb, prev_tick

            h0, w0 = curr_rgb.shape[:2]

            # 1) side-by-side preview modes
            if args.mode in (-2, -1):
                if args.mode == -2:
                    side = np.concatenate((curr_rgb, prev_rgb), axis=1)
                else:
                    side = np.concatenate((curr_rgb, curr_rgb), axis=1)
                writer.write_frame(side)
                prev_rgb = curr_rgb
                return

            # 2) face detection (at 32-multiple)
            if do_faces:
                h1, w1 = round_up32(h0), round_up32(w0)
                det_rgb = cv2.resize(curr_rgb,
                                     (w1, h1),
                                     interpolation=cv2.INTER_LINEAR)
                boxes, scores = face_detector.detect(det_rgb)
                # scale boxes back
                sx, sy = w0 / w1, h0 / h1
                boxes = [
                    (int(x0*sx), int(y0*sy), int(x1*sx), int(y1*sy))
                    for (x0,y0,x1,y1) in boxes
                ]
            else:
                boxes = []

            # 3) depth pass at native resolution
            d_bytes = depth_model.process(curr_rgb, w0, h0)
            d_nat   = np.frombuffer(d_bytes, np.float32).reshape(h0, w0)
            if do_filter:
                d_nat = depth_filter.filter(d_nat, curr_rgb)
            if do_autofocus:
                d_nat = autofocus.remap(d_nat, boxes)
            d_vis = depth_to_gray(d_nat)
            side  = np.concatenate((curr_rgb, d_vis), axis=1)
            writer.write_frame(side)

            # 4) timing
            tick   = cv2.getTickCount()
            dt_ms   = (tick - prev_tick) / cv2.getTickFrequency() * 1000.0
            fps_meas = (1000.0 / dt_ms) if dt_ms > 0 else 0.0
            prev_tick = tick

            # 5) GUI update
            if gui:
                gui.addBuffer("color", curr_rgb)
                gui.addBuffer("depth", d_vis)
                if do_faces:
                    gui.setFaces("color", boxes)

                gui.addTimeSeriesData(
                    "fps",
                    fps_meas,
                    min_val=0.0,
                    max_val=frames.fps,
                    mode=0
                )
                gui.addTimeSeriesData(
                    "ms/frame",
                    dt_ms,
                    min_val=0.0,
                    max_val=dt_ms*2,
                    mode=1
                )
                gui._render()

            prev_rgb = curr_rgb

        with tqdm(total=total, unit='frame', desc=base) as bar:
            run_frame(prev_rgb); bar.update(1)
            for _, rgb in it:
                run_frame(rgb); bar.update(1)

        writer.release()

    try:
        if os.path.isdir(args.input):
            vids = sorted(
                [os.path.join(args.input, f)
                 for f in os.listdir(args.input)
                 if os.path.splitext(f)[1].lower()
                    in {'.mp4','.mov','.avi','.mkv','.webm','.flv','.mpg','.mpeg'}],
                key=os.path.getsize
            )
            for vid in vids:
                process_video(vid)
        else:
            process_video(args.input)
    except KeyboardInterrupt:
        print("\n>>> Exiting on user request.")
        sys.exit(1)
