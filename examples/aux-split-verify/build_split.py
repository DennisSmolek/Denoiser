#!/usr/bin/env python3
"""Build split-graph artifacts (both precisions) for the aux workaround.

Outputs into public/models/ for BOTH fp32 and fp16 of rt_hdr_calb_cnrm:
  full[.fp16].onnx  — the original model (baseline + WASM reference)
  tail[.fp16].onnx  — enc_conv1..output; inputs [enc_conv0_relu6_2, input]
  enc0[.fp16].bin   — enc_conv0 weights f32 OIHW [32,9,3,3] then bias [32]
                      (f32 even for the fp16 model; the kernel accumulates in f32)

Usage: <onnx-venv-python> build_split.py
"""
import os
import shutil
import onnx
from onnx import numpy_helper
from onnx.utils import extract_model

MODELS = "/Users/dex/Documents/GitHub/homefig/Denoiser/packages/denoiser/models"
OUT = "public/models"
SPLIT = "enc_conv0_relu6_2"
os.makedirs(OUT, exist_ok=True)

for suffix in ["", ".fp16"]:
    name = f"rt_hdr_calb_cnrm{suffix}"
    src = f"{MODELS}/{name}.onnx"
    m = onnx.load(src)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    W = init["enc_conv0_W"].astype("float32")   # (32,9,3,3)
    B = init["enc_conv0_B"].astype("float32")
    assert float(init["relu6_min"]) == 0.0 and float(init["relu6_max"]) == 6.0
    with open(f"{OUT}/enc0{suffix}.bin", "wb") as f:
        f.write(W.tobytes()); f.write(B.tobytes())
    extract_model(src, f"{OUT}/tail{suffix}.onnx", [SPLIT, "input"], ["output"])
    shutil.copyfile(src, f"{OUT}/full{suffix}.onnx")
    print(f"{name}: enc0 W{W.shape}+B{B.shape}, tail + full written")
