#!/usr/bin/env python3
# DepthVideo.py – RGB | Depth side-by-side writer with GUI abstraction and per-step timing
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
from TextDepthFilter    import TextDepthFilter
from DetectFace         import FaceDetector
from GUI                import GUI

# bring in your Frame + history helper
from Frame              import Frame, FrameHistory

# Mapping from compile level int to string
_COMPILE_LEVEL_MAP = {
    0: 'fast',   # fastest compile
    1: 'mid',    # medium compile
    2: 'best'    # highest optimization
}

class DepthVideoProcessor:
    @staticmethod
    def round_up32(x: int) -> int:
        return ((x + 31) // 32) * 32

    @staticmethod
    def depth_to_gray(d: np.ndarray) -> np.ndarray:
        mn, mx = float(d.min()), float(d.max())
        if mx - mn < 1e-6:
            mx = mn + 1e-6
        g = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)

    def __init__(self, args):
        self.args        = args
        self.use_nvenc   = not args.cpu
        self.show_prev   = not (args.no_preview or args.skip_ui)
        self.max_frames  = args.max_frames

        self.gui = GUI(window_name='Depth Preview', window_size=(2560,1440), fps=30) if self.show_prev else None

        # Build model_id based on user choice
        size_str = 'Small' if args.model_size.lower() == 'small' else 'Large'
        model_id = f"xingyang1/Distill-Any-Depth-{size_str}-hf"

        # Map compile_level integer to string
        compile_level = _COMPILE_LEVEL_MAP.get(args.compile_level, 'fast')

        self.depth_model = DepthGenerator(
            model_id=model_id,
            compile_level=compile_level,
            use_amp=True,
            gui=self.gui
        )

        # Face detector if mode >= 2
        self.face_detector = (
            FaceDetector(device='cuda', gui=self.gui)
            if args.mode >= 2 else None
        )

        # Text-depth filter if mode >= 4
        self.text_filter = (
            TextDepthFilter(gui=self.gui)
            if args.mode >= 4 else None
        )

        # placeholders; actual filter & history created in process_video()
        self.depth_filter   = None
        self.depth_history  = None
        self.autofocus      = None
        self.writer         = None

        # Timing for ETA
        self.start_tick       = None
        self.frames_total     = 0
        self.frames_processed = 0

    def run_frame(self, curr_rgb):
        freq = cv2.getTickFrequency()
        t_frame_start = cv2.getTickCount()

        # 1) Preview-only modes
        if self.args.mode in (-2, -1):
            side = np.concatenate(
                (curr_rgb,
                 self.prev_rgb if self.args.mode == -2 else curr_rgb),
                axis=1
            )
            t_write = cv2.getTickCount()
            self.writer.write_frame(side)
            if self.gui:
                ms_write = (cv2.getTickCount() - t_write) / freq * 1000.0
                self.gui.addTimeSeriesData('ms/write', ms_write, mode=0)
                # final GUI render
                self.gui.addBuffer('color', curr_rgb)
                self.gui.addBuffer('depth', side[:, :curr_rgb.shape[1]])
                self.gui._render()
            self.prev_rgb = curr_rgb
            return

        # 2) Face detection
        boxes = []
        if self.do_faces:
            t_face = cv2.getTickCount()
            h0, w0 = curr_rgb.shape[:2]
            h1, w1 = self.round_up32(h0), self.round_up32(w0)
            det_rgb = cv2.resize(curr_rgb, (w1, h1), interpolation=cv2.INTER_LINEAR)
            boxes, _ = self.face_detector.detect(det_rgb)
            sx, sy = w0 / w1, h0 / h1
            boxes = [(int(x0*sx), int(y0*sy), int(x1*sx), int(y1*sy))
                     for x0, y0, x1, y1 in boxes]
            if self.gui:
                ms = (cv2.getTickCount() - t_face) / freq * 1000.0
                self.gui.addTimeSeriesData('ms/face', ms, mode=0)

        # 3) Depth estimation
        t_depth = cv2.getTickCount()
        h0, w0 = curr_rgb.shape[:2]
        d_bytes = self.depth_model.process(curr_rgb, w0, h0)
        if self.gui:
            ms = (cv2.getTickCount() - t_depth) / freq * 1000.0
            self.gui.addTimeSeriesData('ms/depth', ms, mode=0)
        d_nat = np.frombuffer(d_bytes, np.float32).reshape(h0, w0)

        # 4) Depth filter (now using Frame + FrameHistory)
        if self.do_filter:
            t_filter = cv2.getTickCount()
            # wrap into Frame, apply filter, store history
            frame = Frame(curr_rgb)
            frame.d_nat = d_nat
            filtered = self.depth_filter.filter(frame.d_nat, curr_rgb)
            frame.d_filtered = filtered
            self.depth_history.append(frame)
            d_nat = filtered
            if self.gui:
                ms = (cv2.getTickCount() - t_filter) / freq * 1000.0
                self.gui.addTimeSeriesData('ms/filter', ms, mode=0)

        # 5) Autofocus
        if self.do_autofocus:
            t_auto = cv2.getTickCount()
            d_nat = self.autofocus.remap(d_nat, boxes)
            if self.gui:
                ms = (cv2.getTickCount() - t_auto) / freq * 1000.0
                self.gui.addTimeSeriesData('ms/autofocus', ms, mode=0)

        # 6) Text filter
        if self.do_text:
            t_text = cv2.getTickCount()
            d_nat = self.text_filter.filter(d_nat, curr_rgb)
            if self.gui:
                ms = (cv2.getTickCount() - t_text) / freq * 1000.0
                self.gui.addTimeSeriesData('ms/text', ms, mode=0)

        # 7) Write output frame
        d_vis = self.depth_to_gray(d_nat)
        side  = np.concatenate((curr_rgb, d_vis), axis=1)
        t_write = cv2.getTickCount()
        self.writer.write_frame(side)
        if self.gui:
            ms = (cv2.getTickCount() - t_write) / freq * 1000.0
            self.gui.addTimeSeriesData('ms/write', ms, mode=0)

        # 8) Frame-level metrics & GUI
        if self.gui:
            t_end = cv2.getTickCount()
            ms_frame = (t_end - t_frame_start) / freq * 1000.0
            fps = 1000.0 / ms_frame if ms_frame > 0 else 0.0
            self.gui.addTimeSeriesData('ms/frame', ms_frame, mode=0)
            self.gui.addTimeSeriesData('fps', fps, mode=0)

            self.frames_processed += 1
            elapsed = (t_end - self.start_tick) / freq
            remaining = self.frames_total - self.frames_processed
            eta = elapsed / max(self.frames_processed, 1) * remaining
            self.gui.addTimeSeriesData('eta_s', eta, mode=0)

            self.gui.addBuffer('color', curr_rgb)
            self.gui.addBuffer('depth', d_vis)
            if self.do_faces:
                self.gui.setFaces('color', boxes)
            self.gui._render()

        # update for next frame
        self.prev_rgb = curr_rgb

    def process_video(self, path_in):
        self.do_filter    = self.args.mode >= 1
        self.do_faces     = self.args.mode >= 2
        self.do_autofocus = self.args.mode >= 3
        self.do_text      = self.args.mode >= 4

        self.frames = VideoFrameGenerator(
            path_in,
            process_length=self.max_frames,
            max_res=self.args.max_res
        )
        it = iter(self.frames)
        try:
            _, self.prev_rgb = next(it)
        except StopIteration:
            return

        H, W = self.prev_rgb.shape[:2]
        base = os.path.splitext(os.path.basename(path_in))[0]
        outdir = './output'
        os.makedirs(outdir, exist_ok=True)
        out_fp = os.path.join(outdir, f'{base}_rgbd.mp4')
        print(f"\nProcessing '{path_in}' → '{out_fp}'")

        # instantiate depth filter + its history
        if self.do_filter:
            self.depth_filter = DepthFilter(
                device='cuda',
                fps=self.frames.fps,
                gui=self.gui
            )
            # history window matches filter.window
            self.depth_history = FrameHistory(maxlen=self.depth_filter.window)

        # autofocus
        if self.do_autofocus:
            self.autofocus = DepthAutoFocus(
                fps=self.frames.fps,
                target=0.5,
                gui=self.gui
            )

        # video writer
        self.writer = FFMPEGVideoWriter(
            output_file=out_fp,
            fps=self.frames.fps,
            width=W*2, height=H,
            audio_file=path_in,
            debug=False,
            use_nvenc=self.use_nvenc
        )

        total = get_video_frame_count(path_in) or int(self.frames.frame_count)
        if self.max_frames > 0 and total > self.max_frames:
            total = self.max_frames
        self.frames_total     = total
        self.start_tick       = cv2.getTickCount()
        self.frames_processed = 0

        with tqdm(total=total, unit='frame', desc=base) as bar:
            self.run_frame(self.prev_rgb)
            bar.update(1)
            for _, rgb in it:
                self.run_frame(rgb)
                bar.update(1)

        self.writer.release()

    def run(self):
        if os.path.isdir(self.args.input):
            vids = sorted(
                [os.path.join(self.args.input, f)
                 for f in os.listdir(self.args.input)
                 if os.path.splitext(f)[1].lower() in {
                     '.mp4','.mov','.avi','.mkv','.webm',
                     '.flv','.mpg','.mpeg'
                 }],
                key=os.path.getsize
            )
            for vid in vids:
                self.process_video(vid)
        else:
            self.process_video(self.args.input)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='RGB + Depth video generator')
    p.add_argument('input',
                   help='input file or folder')
    p.add_argument('--cpu',
                   action='store_true',
                   help='use libx264 instead of NVENC')
    p.add_argument('--no_preview',
                   action='store_true',
                   help='disable preview window')
    p.add_argument('--skip_ui', '-q',
                   action='store_true',
                   help='alias for --no_preview')
    p.add_argument(
        '--mode', type=int,
        choices=[-2, -1, 0, 1, 2, 3, 4],
        default=4,
        help=(
            '-2=delay-right, -1=same-frame, 0=depth, '
            '1=+filter, 2=+faces, 3=+autofocus, 4=+text-filter'
        )
    )
    p.add_argument('--max_res',
                   type=int, default=512,
                   help='maximum frame height (preserves aspect ratio)')
    p.add_argument('--max_frames',
                   type=int, default=-1,
                   help='maximum number of frames to process (default: all)')
    p.add_argument('--model_size',
                   choices=['small','large'], default='small',
                   help='choose depth model size (small or large)')
    p.add_argument('--compile_level',
                   type=int, choices=[0,1,2], default=0,
                   help='compile optimization level: 0=fast,1=mid,2=best')
    args = p.parse_args()

    processor = DepthVideoProcessor(args)
    try:
        processor.run()
    except KeyboardInterrupt:
        print("\n>>> Exiting on user request.")
        sys.exit(1)
