#!/usr/bin/env python3
"""Split an OIDN aux U-Net at enc_conv0 into head + tail ONNX models.

head_enc0.onnx : input(9) -> enc_conv0_relu6_2(32)      (Conv + relu6 only)
tail.onnx      : [enc_conv0_relu6_2(32), input(9)] -> output(3)

The tail takes TWO inputs: the enc_conv0 feature map AND the raw input, because
the OIDN U-Net feeds the raw input into a late decoder skip (concat_38 ->
dec_conv1a). See README.md.

Usage: ./.venv/bin/python make_split.py [src.onnx]
Default src: ../../packages/denoiser/models/rt_hdr_calb_cnrm.onnx
"""
import sys
import shutil
import onnx
from onnx.utils import extract_model

SRC = sys.argv[1] if len(sys.argv) > 1 else "../../packages/denoiser/models/rt_hdr_calb_cnrm.onnx"
OUT = "models"
SPLIT = "enc_conv0_relu6_2"  # output of enc_conv0's Conv+Clip(relu6)

m = onnx.load(SRC)
c1 = next(t for t in m.graph.initializer if t.name == "enc_conv0_W").dims[0]
print(f"src={SRC}  enc_conv0 out channels={c1}")

# head: just enc_conv0 (Conv + relu6)
extract_model(SRC, f"{OUT}/head_enc0.onnx", ["input"], [SPLIT])
# tail: everything after, needs the feature map AND the raw input (dec_conv1a skip)
extract_model(SRC, f"{OUT}/tail.onnx", [SPLIT, "input"], ["output"])
# full model, for the in-browser baseline + reference
shutil.copyfile(SRC, f"{OUT}/full.onnx")

t = onnx.load(f"{OUT}/tail.onnx")
print("tail inputs:", [i.name for i in t.graph.input])
print("wrote models/head_enc0.onnx, models/tail.onnx, models/full.onnx")
