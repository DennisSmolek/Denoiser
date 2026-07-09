#!/usr/bin/env python3
"""Generate aux split-graph artifacts for ALL 9ch cleanAux OIDN models.

For each model, the ORT-web WebGPU Conv bug lives in the FIRST conv that reduces
the raw >3-channel graph input (see tools/ort-webgpu-aux-repro). We split the
graph right after that conv's relu6 and ship:
  <name>.tail.onnx  — the rest of the net; two inputs [<split>, input]
  <name>.enc0.bin   — that first conv's weights, f32 OIHW [COUT,CIN,3,3] then
                      bias [COUT] (f32 even for fp16 models; the engine kernel
                      accumulates in f32).

The first conv is found generically (base/small call it enc_conv0, large calls
it enc_conv1a) as the unique Conv whose data input is the graph input 'input'.

Each split is CPU-verified faithful: full(x) == head(x)+tail(...) so extraction
never drops the input skip. Run with the onnx venv:
  tools/onnx-convert/.venv/bin/python tools/aux-split-artifacts/generate.py \
      [models_dir] [out_dir]
"""
import sys
import os
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.utils import extract_model
import onnxruntime as ort

MODELS = sys.argv[1] if len(sys.argv) > 1 else "/Users/dex/Documents/GitHub/homefig/Denoiser/packages/denoiser/models"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

names = sorted(f[:-5] for f in os.listdir(MODELS)
               if "calb_cnrm" in f and f.endswith(".onnx") and ".tail." not in f)


def first_input_conv(g):
    """The unique Conv whose DATA input is the graph input 'input'."""
    convs = [n for n in g.node if n.op_type == "Conv" and n.input[0] == "input"]
    assert len(convs) == 1, f"expected 1 input-conv, got {[n.name for n in convs]}"
    return convs[0]


def relu6_out(g, conv_out):
    """The Clip(relu6) output fed by conv_out — our split tensor."""
    clip = next(n for n in g.node if n.op_type == "Clip" and n.input[0] == conv_out)
    return clip.output[0]


def run(model_path, feeds):
    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)[0]


ok = 0
for name in names:
    src = f"{MODELS}/{name}.onnx"
    m = onnx.load(src)
    g = m.graph
    init = {t.name: t for t in g.initializer}
    io_f16 = g.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    npdt = np.float16 if io_f16 else np.float32

    conv = first_input_conv(g)
    Wt, Bt = conv.input[1], conv.input[2]
    W = numpy_helper.to_array(init[Wt]).astype("float32")   # (COUT, CIN, 3, 3)
    B = numpy_helper.to_array(init[Bt]).astype("float32")   # (COUT,)
    rmin = float(numpy_helper.to_array(init["relu6_min"])); rmax = float(numpy_helper.to_array(init["relu6_max"]))
    assert W.shape[1] == 9 and W.shape[2:] == (3, 3), W.shape
    assert (rmin, rmax) == (0.0, 6.0), (rmin, rmax)
    split = relu6_out(g, conv.output[0])

    with open(f"{OUT}/{name}.enc0.bin", "wb") as f:
        f.write(W.tobytes()); f.write(B.tobytes())
    tail_path = f"{OUT}/{name}.tail.onnx"
    head_path = f"{OUT}/{name}.head.onnx"  # temp, for verification
    extract_model(src, tail_path, [split, "input"], ["output"])
    extract_model(src, head_path, ["input"], [split])

    # CPU verify the split is faithful: full(x) == head(x) -> tail(feat, x)
    rng = np.random.default_rng(0)
    x = rng.random((1, 9, 64, 64), dtype=np.float32).astype(npdt)
    full = run(src, {"input": x})
    feat = run(head_path, {"input": x})
    tail = run(tail_path, {split: feat, "input": x})
    maxd = float(np.max(np.abs(full.astype(np.float32) - tail.astype(np.float32))))
    os.remove(head_path)
    status = "OK" if maxd < 1e-3 else f"MISMATCH {maxd:.2e}"
    if maxd < 1e-3:
        ok += 1
    print(f"{name:34s} first_conv={conv.name:20s} enc0 {tuple(W.shape)} io={'f16' if io_f16 else 'f32'} faithful={status}")

print(f"\n{ok}/{len(names)} models: artifacts written to {OUT} (<name>.tail.onnx + <name>.enc0.bin)")
