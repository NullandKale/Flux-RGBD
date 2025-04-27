import cv2
import numpy as np
import threading
import subprocess

class VideoFrameGenerator:
    def __init__(self, video_path, process_length=-1, max_res=-1):
        self.video_path = video_path
        self.process_length = process_length
        self.max_res = max_res

        self.cap = cv2.VideoCapture(video_path)
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_index = 0
        
        # If max_res set, scale down frames
        if self.max_res > 0 and max(self.original_height, self.original_width) > self.max_res:
            scale = self.max_res / max(self.original_height, self.original_width)
            self.height = self.ensure_even(round(self.original_height * scale))
            self.width = self.ensure_even(round(self.original_width * scale))
        else:
            self.height = self.original_height
            self.width = self.original_width

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration

        # Current frame index
        fIndex = self.frame_index
        self.frame_index += 1

        # If process_length is set, respect that limit
        if self.process_length != -1 and self.frame_index > self.process_length:
            self.cap.release()
            raise StopIteration

        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if self.max_res > 0 and max(self.original_height, self.original_width) > self.max_res:
            frame = cv2.resize(frame, (self.width, self.height))

        # Return both frame_index and the frame
        return fIndex, frame

    def ensure_even(self, value):
        return value if value % 2 == 0 else value + 1

def get_video_frame_count(video_path):
    """
    Get accurate frame count using ffprobe
    Returns frame count as integer, or None if unavailable
    """
    try:
        # Construct ffprobe command
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames', 
            '-of', 'default=nokey=1:noprint_wrappers=1',
            video_path
        ]
        
        # Execute and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            return int(result.stdout.strip())
            
        # Fallback method if standard method fails
        cmd_fallback = [
            'ffprobe',
            '-v', 'error',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-select_streams', 'v:0',
            '-of', 'default=nokey=1:noprint_wrappers=1',
            video_path
        ]
        
        result_fallback = subprocess.run(
            cmd_fallback,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result_fallback.returncode == 0:
            return int(result_fallback.stdout.strip())

    except FileNotFoundError:
        print("ffprobe not found. Please install ffmpeg.")
    except Exception as e:
        print(f"Error getting frame count: {str(e)}")
        
    return None

class FFMPEGVideoWriter:
    def __init__(self,
                 output_file: str,
                 fps: float,
                 width: int,
                 height: int,
                 audio_file: str | None = None,
                 debug: bool = False,
                 use_nvenc: bool = False):
        """
        Writes raw RGB24 frames (via stdin) to a video file using ffmpeg.
        Optionally merges audio from *audio_file*.

        Parameters
        ----------
        output_file : str   Path to the output video
        fps         : float Frames per second
        width       : int   Frame width  (must be even for many codecs)
        height      : int   Frame height (must be even for many codecs)
        audio_file  : str | None  Source file for audio track (optional)
        debug       : bool  If True, print ffmpeg stderr live
        use_nvenc   : bool  If True, encode with **hevc_nvenc**;
                            otherwise **libx264** (H.264)
        """
        self.width = width
        self.height = height
        self.debug = debug

        # ---------- build ffmpeg command -----------------------------
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",                    # stdin = raw frames
        ]

        if audio_file is not None:
            cmd += ["-i", audio_file]

        # Always map video from stdin
        cmd += ["-map", "0:v:0"]

        # Map audio if present; ? makes it optional if no audio stream
        if audio_file is not None:
            cmd += ["-map", "1:a:0?"]

        # ---------- choose video encoder -----------------------------
        if use_nvenc:
            # HEVC (H.265).  Add yuv420p to guarantee compatibility.
            cmd += [
                "-c:v", "hevc_nvenc",
                "-preset", "medium",
                "-cq", "23",
                "-pix_fmt", "yuv420p",
                "-profile:v", "main",
            ]
        else:
            # Software H.264 for maximum compatibility.
            cmd += [
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
            ]

        # ---------- audio & container flags --------------------------
        if audio_file is not None:
            cmd += ["-c:a", "aac", "-ac", "2"]
        # Write moov atom first → thumbnails work immediately
        cmd += ["-movflags", "+faststart"]

        cmd.append(output_file)

        # ---------- launch subprocess -------------------------------
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=2 ** 24,
        )

        # Drain stderr so buffer can’t fill
        self.stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self.stderr_thread.start()

    # -----------------------------------------------------------------
    def _read_stderr(self):
        for raw in self.process.stderr:
            if self.debug:
                print("ffmpeg stderr:", raw.decode("utf-8", errors="ignore").rstrip())

    # -----------------------------------------------------------------
    def write_frame(self, frame: np.ndarray):
        """Write one RGB24 frame (H×W×3, uint8) to ffmpeg stdin."""
        exp = (self.height, self.width, 3)
        if frame.shape != exp:
            raise ValueError(f"frame shape {frame.shape} != expected {exp}")
        if frame.dtype != np.uint8:
            raise ValueError("frame dtype must be uint8")

        if self.process.poll() is not None:
            raise RuntimeError(f"ffmpeg exited early (code {self.process.returncode})")

        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            raise RuntimeError(
                "Broken ffmpeg pipe — enable debug=True to inspect stderr."
            ) from e

    # -----------------------------------------------------------------
    def release(self):
        """Close stdin, join stderr thread, and wait for ffmpeg to exit."""
        if getattr(self.process, "stdin", None) and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except Exception:
                pass

        self.stderr_thread.join(timeout=1)

        if self.process.poll() is None:
            self.process.wait()

    # -----------------------------------------------------------------
    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
