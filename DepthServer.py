#!/usr/bin/env python3
# DepthServer_small_fp32.py – uses DepthGenerator class
# -----------------------------------------------------
import os, time, logging, traceback, numpy as np
from flask import Flask, request, Response, abort, jsonify

from DepthGenerator import DepthGenerator  

# --------------------------------------------------------------------------
gen = DepthGenerator()                        # single global instance
# --------------------------------------------------------------------------

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR); app.logger.disabled = True

seg, first_ts = 0, None
acc = dict.fromkeys(["parse","read","decode","pre","infer","up","proc"], 0.0)

@app.route("/depth_raw", methods=["POST"])
def depth_raw():
    """
    POST raw BGRA/RGBA8 (4 bytes / pixel) in the request body and get
    FP32 depth back.  Dimensions are query params:

        POST /depth_raw?width=…&height=…

    The only copies on the Python side are:
        • buf → NumPy view (no copy)
        • view.copy()  (required because we mutate it on GPU)
    """
    global seg, first_ts, acc

    t0 = time.perf_counter()
    if seg == 0:
        first_ts = t0

    try:
        w = int(request.args["width"])
        h = int(request.args["height"])
    except Exception:
        abort(400, "need width & height query parameters")

    buf = request.get_data()
    if len(buf) != w * h * 4:                         # BGRA8
        abort(400, f"buffer size mismatch ({len(buf)} ≠ {w*h*4})")

    rgba = np.frombuffer(buf, np.uint8).reshape(h, w, 4).copy()

    out_bytes, model_t = gen.process(rgba, w, h)      # FP32 little-endian

    resp = Response(out_bytes, mimetype="application/octet-stream")

    acc["proc"] += (time.perf_counter() - t0) * 1e3
    seg += 1
    if seg >= 60:
        dt = t0 - first_ts
        print(f"[@60] proc={acc['proc']/seg:.2f} ms   fps_roundtrip={seg/dt:.1f}")
        seg, first_ts = 0, None
        acc = dict.fromkeys(acc, 0.0)

    return resp


@app.errorhandler(Exception)
def err(e):
    tb = traceback.format_exc(); print(tb)
    return jsonify(error=str(e), traceback=tb), 500

if __name__ == "__main__":
    print("Depth server ready on 0.0.0.0:5001  (FP32 + CUDA graphs, class refactor)")
    app.run("0.0.0.0", 5001, debug=False, threaded=False, use_reloader=False)
