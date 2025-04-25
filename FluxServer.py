#!/usr/bin/env python3
import os
import gc
import torch
import traceback
from io import BytesIO
from uuid import uuid4
from PIL import Image
from flask import Flask, request, send_file, abort, jsonify

# quantization helpers
from optimum.quanto import quantize, freeze, qfloat8

# Diffusers / FLUX
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

# Transformers
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    T5EncoderModel, T5TokenizerFast,
    pipeline as hf_pipeline
)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

# ------------------------------------------------------------------------------
flush()
print("Loading quantized FLUX pipeline...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16     # run FLUX in fp16 on GPU

REPO     = "black-forest-labs/FLUX.1-schnell"
REVISION = "refs/pr/1"

# Scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    REPO, subfolder="scheduler", revision=REVISION
)

# CLIP encoder / tokenizer
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=DTYPE
)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# Secondary text encoder / tokenizer (quantized)
text_encoder_2 = T5EncoderModel.from_pretrained(
    REPO, subfolder="text_encoder_2", torch_dtype=DTYPE, revision=REVISION
)
tokenizer_2 = T5TokenizerFast.from_pretrained(
    REPO, subfolder="tokenizer_2", revision=REVISION
)

# VAE
vae = AutoencoderKL.from_pretrained(
    REPO, subfolder="vae", torch_dtype=DTYPE, revision=REVISION
)

# Transformer
transformer = FluxTransformer2DModel.from_pretrained(
    REPO, subfolder="transformer", torch_dtype=DTYPE, revision=REVISION
)

# Quantize & freeze
quantize(transformer, weights=qfloat8); freeze(transformer)
quantize(text_encoder_2, weights=qfloat8); freeze(text_encoder_2)

# Build FLUX pipeline
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
).to(DEVICE)

# Try to enable xFormers for memory‑efficient attention
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✔️  xFormers efficient attention enabled")
except Exception:
    print("⚠️  xFormers not available, skipping")

# ------------------------------------------------------------------------------
print("Loading Depth Estimation Pipeline (FP32)...")
depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="xingyang1/Distill-Any-Depth-Small-hf",
    device=0 if DEVICE == "cuda" else -1,
    feature_extractor_use_fast=True
)

# ------------------------------------------------------------------------------
app = Flask(__name__)
generator = torch.Generator(device=DEVICE).manual_seed(12345)
output_folder = os.path.join(os.getcwd(), "generated_images")
os.makedirs(output_folder, exist_ok=True)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt") or abort(400, "Expected JSON: {'prompt': '...'}")
    W, H = data.get("width", 1024), data.get("height", 1024)

    # 1) Generate color image
    try:
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            result = pipe(
                prompt=prompt,
                width=W, height=H,
                num_inference_steps=4,
                generator=generator,
                guidance_scale=3.5,
            )
        color = result.images[0]
    except Exception as e:
        traceback.print_exc()
        abort(500, f"FLUX generation error: {e}")

    # 2) Estimate depth
    try:
        depth_out = depth_pipe(color)
        # handle both dict and list outputs
        if isinstance(depth_out, list):
            depth_img = depth_out[0]["depth"]
        else:
            depth_img = depth_out["depth"]
    except Exception as e:
        traceback.print_exc()
        abort(500, f"Depth estimation error: {e}")

    # 3) Combine side‑by‑side
    combined = Image.new(
        "RGB",
        (color.width + depth_img.width, max(color.height, depth_img.height))
    )
    combined.paste(color, (0, 0))
    combined.paste(depth_img.convert("RGB"), (color.width, 0))

    # 4) Send back PNG
    buf = BytesIO()
    combined.save(buf, format="PNG")
    buf.seek(0)

    # Cleanup
    del result, color, depth_out, depth_img, combined
    flush()

    return send_file(buf, mimetype="image/png")


@app.route("/generate_file", methods=["POST"])
def generate_file():
    data = request.get_json(force=True)
    prompt = data.get("prompt") or abort(400, "Expected JSON: {'prompt': '...'}")
    W, H = data.get("width", 1024), data.get("height", 1024)

    try:
        with torch.autocast(device_type=DEVICE, dtype=DTYPE):
            result = pipe(
                prompt=prompt,
                width=W, height=H,
                num_inference_steps=4,
                generator=generator,
                guidance_scale=3.5,
            )
        color = result.images[0]
    except Exception as e:
        traceback.print_exc()
        abort(500, f"FLUX generation error: {e}")

    try:
        depth_out = depth_pipe(color)
        if isinstance(depth_out, list):
            depth_img = depth_out[0]["depth"]
        else:
            depth_img = depth_out["depth"]
    except Exception as e:
        traceback.print_exc()
        abort(500, f"Depth estimation error: {e}")

    combined = Image.new(
        "RGB",
        (color.width + depth_img.width, max(color.height, depth_img.height))
    )
    combined.paste(color, (0, 0))
    combined.paste(depth_img.convert("RGB"), (color.width, 0))

    filename = f"generated_{uuid4().hex}.png"
    path = os.path.join(output_folder, filename)
    combined.save(path, format="PNG")

    del result, color, depth_out, depth_img, combined
    flush()

    return jsonify({"file_path": path})


@app.errorhandler(Exception)
def on_error(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), getattr(e, "code", 500)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
