#!/usr/bin/env python3
# DepthServer_small_fp32.py – uses DepthGenerator class
# -----------------------------------------------------
import os, time, logging, traceback, numpy as np
from flask import Flask, request, Response, abort, jsonify

from DepthGenerator import DepthGenerator  

# --------------------------------------------------------------------------
gen = DepthGenerator(model_id="xingyang1/Distill-Any-Depth-Small-hf", compile_level="mid")
# --------------------------------------------------------------------------

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR); app.logger.disabled = True


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

    try:
        w = int(request.args["width"])
        h = int(request.args["height"])
    except Exception:
        abort(400, "need width & height query parameters")

    buf = request.get_data()
    if len(buf) != w * h * 4:                         # BGRA8
        abort(400, f"buffer size mismatch ({len(buf)} ≠ {w*h*4})")

    rgba = np.frombuffer(buf, np.uint8).reshape(h, w, 4).copy()

    out_bytes = gen.process(rgba, w, h)      # FP32 little-endian

    resp = Response(out_bytes, mimetype="application/octet-stream")

    return resp


@app.errorhandler(Exception)
def err(e):
    tb = traceback.format_exc(); print(tb)
    return jsonify(error=str(e), traceback=tb), 500

if __name__ == "__main__":
    print("Depth server ready on 0.0.0.0:5001  (FP32 + CUDA graphs, class refactor)")
    app.run("0.0.0.0", 5001, debug=False, threaded=False, use_reloader=False)
