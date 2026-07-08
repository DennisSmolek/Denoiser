#!/usr/bin/env python3
"""Build split-graph artifacts for the aux workaround, for one model.

Outputs into public/models/:
  full.onnx  — the original model (in-browser baseline + WASM reference)
  tail.onnx  — enc_conv1..output; inputs [enc_conv0_relu6_2(32), input(9)]
  enc0.bin   — enc_conv0 weights: f32 OIHW [32,9,3,3] then f32 bias [32],
               little-endian, contiguous (2592 + 32 = 2624 floats)

The WGSL enc_conv0 kernel indexes weights as (co*cin+ci)*9 + ky*3+kx, which is
exactly numpy's C-order for a (COUT,CIN,3,3) array — so .tobytes() drops straight
into the GPU buffer.

Usage: <onnx-venv-python> build_split.py [src.onnx]
"""
import sys
import shutil
import onnx
from onnx import numpy_helper
from onnx.utils import extract_model

DEFAULT_SRC = "/Users/dex/Documents/GitHub/homefig/Denoiser/packages/denoiser/models/rt_hdr_calb_cnrm.onnx"
SRC = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
OUT = "public/models"
SPLIT = "enc_conv0_relu6_2"

import os
os.makedirs(OUT, exist_ok=True)

m = onnx.load(SRC)
init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
W = init["enc_conv0_W"].astype("float32")   # (32, 9, 3, 3)
B = init["enc_conv0_B"].astype("float32")   # (32,)
cout, cin = W.shape[0], W.shape[1]
assert W.shape[2:] == (3, 3), W.shape
assert float(init["relu6_min"]) == 0.0 and float(init["relu6_max"]) == 6.0

with open(f"{OUT}/enc0.bin", "wb") as f:
    f.write(W.tobytes())   # C-order OIHW
    f.write(B.tobytes())
print(f"enc0.bin: W{W.shape} + B{B.shape}  ({W.size + B.size} floats)")

extract_model(SRC, f"{OUT}/tail.onnx", [SPLIT, "input"], ["output"])
shutil.copyfile(SRC, f"{OUT}/full.onnx")
t = onnx.load(f"{OUT}/tail.onnx")
print("tail inputs:", [i.name for i in t.graph.input])
print(f"wrote {OUT}/full.onnx, tail.onnx, enc0.bin  (cin={cin} cout={cout})")
