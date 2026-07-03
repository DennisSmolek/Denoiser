"""Emit conv.onnx (3x3 Conv, Cin->Cout, dynamic H/W) plus the raw weights/bias
so ORT (reference) and the WGSL kernels use identical parameters.
Run with the repo's converter venv (numpy + onnx)."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CIN = COUT = 64
rng = np.random.default_rng(7)
W = (rng.standard_normal((COUT, CIN, 3, 3)) * 0.05).astype(np.float32)
B = (rng.standard_normal(COUT) * 0.1).astype(np.float32)

node = helper.make_node("Conv", ["input", "W", "B"], ["output"],
                        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
graph = helper.make_graph(
    [node], "conv_spike",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, CIN, "height", "width"])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, COUT, "height", "width"])],
    [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, "conv.onnx")
W.tofile("weights.bin")  # OIHW f32
B.tofile("bias.bin")
print(f"wrote conv.onnx ({CIN}->{COUT}), weights.bin, bias.bin")
