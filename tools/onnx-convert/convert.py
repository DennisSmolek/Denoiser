"""Convert OIDN TZA weight blobs into ONNX U-Net models.

Builds the ONNX graph directly from the parsed TZA tensors, mirroring the
TensorFlow graph in packages/denoiser/src/unet.ts 1:1:

    enc_conv0 -> enc_conv1 -> pool1
              -> enc_conv2 -> pool2
              -> enc_conv3 -> pool3
              -> enc_conv4 -> pool4
              -> enc_conv5a -> enc_conv5b (bottleneck)
    up -> concat(pool3) -> dec_conv4a -> dec_conv4b
    up -> concat(pool2) -> dec_conv3a -> dec_conv3b
    up -> concat(pool1) -> dec_conv2a -> dec_conv2b
    up -> concat(input) -> dec_conv1a -> dec_conv1b
    dec_conv0 -> output

Conv = 3x3 same-pad + bias, activation relu6 (Clip[0,6]). Pool = 2x2/stride2
max pool. Upsample = nearest 2x (Resize, asymmetric/floor to match TF
upSampling2d pixel duplication). Concat is along the channel axis (NCHW, axis=1).

Layout is NCHW. By default the input shape is fully dynamic with NAMED free
dims [batch, C, height, width] so the runtime pins them per session via
freeDimensionOverrides (batching, any /16-aligned tile size, whole-frame runs).
--size N emits the old fixed [1, C, N, N] shape instead. All channel counts
are inferred from the weights.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tza import parse_tza

OPSET = 17

# Conv block order, matching unet.ts. None marks where a skip-concat is injected
# before the block (with the named skip source).
ENCODER = ["enc_conv0", "enc_conv1", "enc_conv2", "enc_conv3", "enc_conv4",
           "enc_conv5a", "enc_conv5b"]


def _init(name: str, arr: np.ndarray, fp16: bool) -> onnx.TensorProto:
    if fp16 and arr.dtype == np.float32:
        arr = arr.astype(np.float16)
    return numpy_helper.from_array(arr, name=name)


class GraphBuilder:
    def __init__(self, weights: Dict[str, np.ndarray], fp16: bool, final_activation: str):
        self.w = weights
        self.fp16 = fp16
        self.final_activation = final_activation
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._uid = 0
        # shared clip bounds for relu6
        self.inits.append(_init("relu6_min", np.array(0.0, np.float32), fp16))
        self.inits.append(_init("relu6_max", np.array(6.0, np.float32), fp16))
        self.inits.append(_init("up_scales", np.array([1, 1, 2, 2], np.float32), False))

    def _n(self, tag: str) -> str:
        self._uid += 1
        return f"{tag}_{self._uid}"

    def conv(self, name: str, x: str, activation: bool = True) -> str:
        wname, bname = f"{name}.weight", f"{name}.bias"
        self.inits.append(_init(f"{name}_W", self.w[wname], self.fp16))
        self.inits.append(_init(f"{name}_B", self.w[bname], self.fp16))
        out = self._n(f"{name}_conv")
        self.nodes.append(helper.make_node(
            "Conv", [x, f"{name}_W", f"{name}_B"], [out],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], name=out))
        if not activation:
            return out
        relu = self._n(f"{name}_relu6")
        self.nodes.append(helper.make_node(
            "Clip", [out, "relu6_min", "relu6_max"], [relu], name=relu))
        return relu

    def pool(self, x: str) -> str:
        out = self._n("pool")
        self.nodes.append(helper.make_node(
            "MaxPool", [x], [out], kernel_shape=[2, 2], strides=[2, 2], name=out))
        return out

    def up(self, x: str) -> str:
        out = self._n("up")
        self.nodes.append(helper.make_node(
            "Resize", [x, "", "up_scales"], [out],
            mode="nearest", nearest_mode="floor",
            coordinate_transformation_mode="asymmetric", name=out))
        return out

    def concat(self, a: str, b: str) -> str:
        out = self._n("concat")
        self.nodes.append(helper.make_node("Concat", [a, b], [out], axis=1, name=out))
        return out

    def _build_standard(self) -> str:
        """Standard U-Net, 1:1 with unet.ts. Returns the output conv name."""
        x = self.conv("enc_conv0", "input")
        x = self.conv("enc_conv1", x)
        pool1 = self.pool(x)
        x = self.conv("enc_conv2", pool1); pool2 = self.pool(x)
        x = self.conv("enc_conv3", pool2); pool3 = self.pool(x)
        x = self.conv("enc_conv4", pool3); pool4 = self.pool(x)
        x = self.conv("enc_conv5a", pool4)
        x = self.conv("enc_conv5b", x)
        x = self.conv("dec_conv4a", self.concat(self.up(x), pool3)); x = self.conv("dec_conv4b", x)
        x = self.conv("dec_conv3a", self.concat(self.up(x), pool2)); x = self.conv("dec_conv3b", x)
        x = self.conv("dec_conv2a", self.concat(self.up(x), pool1)); x = self.conv("dec_conv2b", x)
        x = self.conv("dec_conv1a", self.concat(self.up(x), "input")); x = self.conv("dec_conv1b", x)
        self.conv("dec_conv0", x, activation=(self.final_activation == "relu6"))
        return "dec_conv0"

    def _build_large(self) -> str:
        """UNetLarge topology (two convs/stage, 3-conv decoder tail). Returns output conv name."""
        x = self.conv("enc_conv1a", "input"); x = self.conv("enc_conv1b", x); pool1 = self.pool(x)
        x = self.conv("enc_conv2a", pool1); x = self.conv("enc_conv2b", x); pool2 = self.pool(x)
        x = self.conv("enc_conv3a", pool2); x = self.conv("enc_conv3b", x); pool3 = self.pool(x)
        x = self.conv("enc_conv4a", pool3); x = self.conv("enc_conv4b", x); pool4 = self.pool(x)
        x = self.conv("enc_conv5a", pool4); x = self.conv("enc_conv5b", x)
        x = self.conv("dec_conv4a", self.concat(self.up(x), pool3)); x = self.conv("dec_conv4b", x)
        x = self.conv("dec_conv3a", self.concat(self.up(x), pool2)); x = self.conv("dec_conv3b", x)
        x = self.conv("dec_conv2a", self.concat(self.up(x), pool1)); x = self.conv("dec_conv2b", x)
        x = self.conv("dec_conv1a", self.concat(self.up(x), "input")); x = self.conv("dec_conv1b", x)
        self.conv("dec_conv1c", x, activation=(self.final_activation == "relu6"))
        return "dec_conv1c"

    def build(self, in_channels: int, height, width) -> onnx.ModelProto:
        """height/width may be ints (static) or dim_param names (dynamic).

        Dynamic models use named free dims ("batch"/"height"/"width") so the
        runtime can pin them per session via freeDimensionOverrides — one
        artifact serves any tile size and batch count (H, W must be multiples
        of 16 for the four pool/upsample round-trips to align).
        """
        elem = TensorProto.FLOAT16 if self.fp16 else TensorProto.FLOAT
        batch = 1 if isinstance(height, int) else "batch"
        inp = helper.make_tensor_value_info("input", elem, [batch, in_channels, height, width])

        if "enc_conv1a.weight" in self.w:
            out_conv = self._build_large()   # UNetLarge topology
        else:
            out_conv = self._build_standard()  # mirrors unet.ts

        out_channels = self.w[f"{out_conv}.weight"].shape[0]
        out = helper.make_tensor_value_info("output", elem, [batch, out_channels, height, width])

        # rename last node's output to "output"
        self.nodes[-1].output[0] = "output"

        graph = helper.make_graph(self.nodes, "oidn_unet", [inp], [out], self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
        model.ir_version = 10  # compatible with onnxruntime-web 1.27
        onnx.checker.check_model(model)
        return model


def convert(tza_path: str, out_path: str, height, width,
            fp16: bool, final_activation: str) -> onnx.ModelProto:
    with open(tza_path, "rb") as f:
        weights = parse_tza(f.read())
    first = "enc_conv1a" if "enc_conv1a.weight" in weights else "enc_conv0"
    in_channels = weights[f"{first}.weight"].shape[1]
    builder = GraphBuilder(weights, fp16, final_activation)
    model = builder.build(in_channels, height, width)
    onnx.save(model, out_path)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert OIDN TZA weights to ONNX U-Nets")
    ap.add_argument("inputs", nargs="+", help="TZA file(s)")
    ap.add_argument("-o", "--outdir", default="packages/denoiser/models")
    ap.add_argument("--size", type=int, default=0,
                    help="fixed square tile size; 0 (default) = dynamic batch/height/width free dims")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--final-activation", choices=["relu6", "none"], default="relu6",
                    help="dec_conv0 activation; relu6 matches current unet.ts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for tza in args.inputs:
        base = os.path.splitext(os.path.basename(tza))[0]
        suffix = ".fp16.onnx" if args.fp16 else ".onnx"
        out = os.path.join(args.outdir, base + suffix)
        hw = (args.size, args.size) if args.size else ("height", "width")
        model = convert(tza, out, hw[0], hw[1], args.fp16, args.final_activation)
        topo = "large" if any("enc_conv1a" in n for n in (init.name for init in model.graph.initializer)) else "std"
        size_mb = os.path.getsize(out) / 1e6
        print(f"{base:28s} [{topo:5s}] -> {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
