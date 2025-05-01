import cv2
import numpy as np
import math
from collections import deque
import colorsys
import sys

# -----------------------------------------------------------------------------
# GUI: tiles buffers, draws faces, shows unlimited time‑series (text + graph)
# -----------------------------------------------------------------------------

class Renderer:
    def draw(self, canvas: np.ndarray, gui: "GUI"):
        raise NotImplementedError

# -----------------------------------------------------------------------------
# 1) Image‑buffer tiler
# -----------------------------------------------------------------------------
class BufferRenderer(Renderer):
    """Tiles image buffers and draws face boxes *after* scaling so alignment is correct."""
    def draw(self, canvas: np.ndarray, gui: "GUI"):
        w_win, h_win = gui.window_size
        img_h = h_win * 3 // 4  # top 75% for images
        
        # 1) Collect names
        all_names = list(gui.buffers.keys())
        unnumbered = []
        numbered   = []
        for name in all_names:
            parts = name.split(":", 1)
            if len(parts) == 2 and parts[0].isdigit():
                numbered.append((int(parts[0]), name))
            else:
                unnumbered.append(name)
        
        # 2) Sort the numbered ones by their integer prefix
        numbered.sort(key=lambda x: x[0])
        numbered = [name for _, name in numbered]
        
        # 3) Final ordering: unnumbered first, then numbered
        names = unnumbered + numbered

        if not names:
            return

        # layout
        n = len(names)
        cols = n if n <= 2 else math.ceil(math.sqrt(n))
        rows = 1 if n <= 2 else math.ceil(n / cols)
        tile_w, tile_h = w_win // cols, img_h // rows

        for idx, name in enumerate(names):
            src = gui.buffers[name]
            ih, iw = src.shape[:2]
            scale = min(tile_w / iw, tile_h / ih)
            rt_w, rt_h = int(iw * scale), int(ih * scale)

            # resize
            thumb = cv2.resize(src, (rt_w, rt_h), interpolation=cv2.INTER_AREA)

            # draw faces
            for (x0, y0, x1, y1) in gui.face_overlays.get(name, []):
                cv2.rectangle(
                    thumb,
                    (int(x0 * scale), int(y0 * scale)),
                    (int(x1 * scale), int(y1 * scale)),
                    (0, 255, 0), 2,
                )

            # label
            tsz, _ = cv2.getTextSize(name, gui.font, gui.font_scale, gui.font_thickness)
            cv2.rectangle(thumb, (0, 0), (tsz[0] + 6, tsz[1] + 6), (0, 0, 0), cv2.FILLED)
            cv2.putText(thumb, name, (3, tsz[1] + 3),
                        gui.font, gui.font_scale, (255, 255, 255), gui.font_thickness)

            # blit
            r, c = divmod(idx, cols)
            x0 = c * tile_w + (tile_w - rt_w) // 2
            y0 = r * tile_h + (tile_h - rt_h) // 2
            canvas[y0:y0 + rt_h, x0:x0 + rt_w] = thumb

# ----------------------------------------------------------------------------- 
# 2) Text-series block (mode 0) with proper vertical spacing 
# -----------------------------------------------------------------------------
class TextSeriesRenderer(Renderer):
    PAD_TOP      = 8
    PAD_RIGHT    = 10
    LINE_SPACING = 10

    def draw(self, canvas: np.ndarray, gui: "GUI") -> int:
        w_win, h_win = gui.window_size
        graph_h      = h_win // 4

        pad_top   = int(self.PAD_TOP    * gui.font_scale)
        pad_right = int(self.PAD_RIGHT  * gui.font_scale)
        spacing   = int(self.LINE_SPACING * gui.font_scale)

        # measure one line height
        sample_txt = "Hg"
        (_, th), _ = cv2.getTextSize(
            sample_txt,
            gui.font,
            gui.font_scale,
            gui.font_thickness
        )
        line_h = th + spacing

        # start baseline one text-height down
        y = pad_top + th

        items = [(n, ts) for n, ts in gui.time_series.items() if ts["mode"] == 0]
        for name, ts in items:
            vals = list(ts["values"])
            if not vals:
                continue

            cur  = vals[-1]
            avg1 = sum(vals[-gui.fps:]) / max(1, min(len(vals), gui.fps))
            avg5 = sum(vals[-gui.fps * 5:]) / max(1, min(len(vals), gui.fps * 5))

            txt = f"{name}: {cur:.2f}  1s {avg1:.2f}  5s {avg5:.2f}"
            (tw, _), _ = cv2.getTextSize(
                txt,
                gui.font,
                gui.font_scale,
                gui.font_thickness
            )

            x = w_win - pad_right - tw
            cv2.putText(
                canvas, txt, (x, y),
                gui.font, gui.font_scale,
                (255, 255, 255), gui.font_thickness
            )
            y += line_h

        # return the y-coordinate where the graph should begin (plus a little padding)
        return y + pad_top

# ----------------------------------------------------------------------------- 
# 3) Graph block (mode 1): labels at top-left, plot in bottom 1/4 
# -----------------------------------------------------------------------------
class GraphSeriesRenderer(Renderer):
    BASE_PAD     = 8
    X_MARGIN     = 10
    LINE_SPACING = 10

    def draw(self, canvas: np.ndarray, gui: "GUI", *_):
        w_win, h_win = gui.window_size

        pad     = int(self.BASE_PAD     * gui.font_scale)
        margin  = int(self.X_MARGIN     * gui.font_scale)
        spacing = int(self.LINE_SPACING * gui.font_scale)

        legend_items = [n for n, ts in gui.time_series.items() if ts["mode"] == 1]
        if not legend_items:
            return

        # measure legend line height
        sample_txt = f"{legend_items[0]}: 00.00"
        (_, lh), _ = cv2.getTextSize(
            sample_txt,
            gui.font,
            gui.font_scale,
            gui.font_thickness
        )
        line_h = lh + spacing

        # 1) Draw all labels starting one line-height down
        y = pad + lh
        for name in legend_items:
            latest = gui.time_series[name]["values"][-1]
            cv2.putText(
                canvas,
                f"{name}: {latest:.2f}",
                (margin, y),
                gui.font,
                gui.font_scale,
                gui._color_for_series(name),
                gui.font_thickness
            )
            y += line_h

        # 2) Reserve bottom 1/4 for plotting
        graph_h  = h_win // 4
        graph_y0 = h_win - graph_h
        plot_top    = graph_y0 + pad
        plot_bottom = h_win - pad
        plot_h      = plot_bottom - plot_top
        plot_w      = w_win

        # background
        cv2.rectangle(canvas,
                      (0, graph_y0),
                      (w_win, h_win),
                      (30, 30, 30),
                      cv2.FILLED)

        # 3) Plot each series line scaled into [plot_top…plot_bottom]
        for name in legend_items:
            ts = gui.time_series[name]
            vs = list(ts["values"])
            if len(vs) < 2:
                continue

            r_min, r_max = ts["min"], ts["max"]
            r_range = r_max - r_min
            if abs(r_range) < 1e-6:
                continue

            N   = len(vs)
            col = gui._color_for_series(name)
            pts = []
            for i, v in enumerate(vs):
                x = int(i * (plot_w / (N - 1)))
                norm = (v - r_min) / r_range
                yv   = int(plot_top + (1.0 - norm) * plot_h)
                pts.append((x, yv))

            for p, q in zip(pts, pts[1:]):
                cv2.line(canvas, p, q, col, 2)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
class GUI:
    def __init__(
        self,
        window_name: str = "Preview",
        window_size: tuple[int, int] = (2560, 720),
        fps: int = 30,
        font_scale: float = 0.7,
        font_thickness: int = 2,
        swap_brg: bool = False,  # NOTE: swap R↔G (RGB→BRG)
    ):
        self.window_name = window_name
        self.window_size = window_size
        self.fps = fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.swap_brg = swap_brg

        self.base_colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0),
            (0, 255, 255), (255, 255, 0), (255, 0, 255),
        ]
        self.buffers, self.face_overlays, self.time_series = {}, {}, {}
        self.renderers = [BufferRenderer(), TextSeriesRenderer(), GraphSeriesRenderer()]

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
        cv2.resizeWindow(self.window_name, *self.window_size)

    # public API
    def addBuffer(self, name: str, img: np.ndarray):
        if self.swap_brg:
            buf = img[..., [2, 0, 1]]
        else:
            buf = img[..., ::-1]
        self.buffers[name] = buf.copy()

    def setFaces(self, name: str, boxes):
        self.face_overlays[name] = list(boxes)

    def addTimeSeriesData(self, name, data, *, min_val=0.0, max_val=1.0, mode=0):
        ts = self.time_series.setdefault(name, dict(values=deque(maxlen=self.fps * 5), min=min_val, max=max_val, mode=mode))
        ts.update(min=min_val, max=max_val, mode=mode)
        ts['values'].append(float(data))

    def clear(self):
        self.buffers.clear()
        self.face_overlays.clear()
        self.time_series.clear()
        self._show_canvas(np.zeros((self.window_size[1], self.window_size[0], 3), np.uint8))

    def _render(self):
        canvas = np.zeros((self.window_size[1], self.window_size[0], 3), np.uint8)

        # start with 0; TextSeriesRenderer.draw() will return the true graph_y0
        graph_y0 = 0

        for r in self.renderers:
            if isinstance(r, BufferRenderer):
                r.draw(canvas, self)
            elif isinstance(r, TextSeriesRenderer):
                # this now computes its own PAD_TOP/PAD_BOTTOM and returns the correct graph_y0
                graph_y0 = r.draw(canvas, self)
            elif isinstance(r, GraphSeriesRenderer):
                r.draw(canvas, self, graph_y0)

        self._show_canvas(canvas)

    # internal
    def _show_canvas(self, canvas: np.ndarray):
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            raise KeyboardInterrupt
        # if the user closed the window, exit immediately
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            sys.exit(0)

    def _color_for_series(self, key: str):
        keys = sorted(self.time_series.keys())
        idx = keys.index(key)
        if idx < len(self.base_colors):
            return self.base_colors[idx]
        hue = (idx * 37) % 360
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 0.8, 1)]
        return (b, g, r)
