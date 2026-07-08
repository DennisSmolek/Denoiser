#!/usr/bin/env python3
"""Generate aux split-graph artifacts for the models this example may load.

For each model: <name>.tail.onnx (enc_conv1..output, inputs [enc_conv0_relu6_2,
input]) and <name>.enc0.bin (f32 OIHW enc_conv0 weights then bias). Served under
/models by vite.config.ts (splitDir is checked before the real models dir).

The example loads rt_hdr_calb_cnrm_small at fast quality and rt_hdr_calb_cnrm at
balanced. Run with the onnx venv:
  ../../../tools/onnx-convert/.venv/bin/python build.py
"""
import os
import onnx
from onnx import numpy_helper
from onnx.utils import extract_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.normpath(os.path.join(HERE, "../../../packages/denoiser/models"))
NAMES = ["rt_hdr_calb_cnrm_small", "rt_hdr_calb_cnrm"]

for name in NAMES:
    src = f"{MODELS}/{name}.onnx"
    if not os.path.exists(src):
        print(f"skip {name} (no {src})"); continue
    m = onnx.load(src)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    W = init["enc_conv0_W"].astype("float32"); B = init["enc_conv0_B"].astype("float32")
    assert float(init["relu6_min"]) == 0.0 and float(init["relu6_max"]) == 6.0
    with open(f"{HERE}/{name}.enc0.bin", "wb") as f:
        f.write(W.tobytes()); f.write(B.tobytes())
    extract_model(src, f"{HERE}/{name}.tail.onnx", ["enc_conv0_relu6_2", "input"], ["output"])
    print(f"{name}: enc0 W{W.shape}+B{B.shape}, tail written")
