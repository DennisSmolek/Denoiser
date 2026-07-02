"""Parity check: NumPy reference of the unet.ts graph vs the exported ONNX run
through onnxruntime. Validates that the ONNX export reproduces the intended
graph semantics (OIHW conv, same-pad, relu6, 2x2 maxpool, nearest 2x upsample,
channel-axis skip concat). Uses a small input for speed.
"""
from __future__ import annotations

import sys

import numpy as np
import onnxruntime as ort

from tza import parse_tza
from convert import convert


def conv2d_same(x, w, b):
    # x: (1,C,H,W)  w: (O,I,3,3) OIHW  b: (O,)
    O, I, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    _, _, H, W = x.shape
    # im2col
    cols = np.zeros((I * kh * kw, H * W), np.float32)
    idx = 0
    for c in range(I):
        for i in range(kh):
            for j in range(kw):
                cols[idx] = xp[0, c, i:i + H, j:j + W].reshape(-1)
                idx += 1
    wm = w.reshape(O, -1)
    out = (wm @ cols).reshape(O, H, W) + b.reshape(O, 1, 1)
    return out[None].astype(np.float32)


def relu6(x):
    return np.clip(x, 0.0, 6.0)


def maxpool2(x):
    _, C, H, W = x.shape
    return x.reshape(1, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def up2(x):
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)


def reference(weights, inp):
    def cv(name, x, act=True):
        y = conv2d_same(x, weights[f"{name}.weight"], weights[f"{name}.bias"])
        return relu6(y) if act else y

    x = cv("enc_conv0", inp)
    x = cv("enc_conv1", x)
    pool1 = maxpool2(x)
    x = cv("enc_conv2", pool1); pool2 = maxpool2(x)
    x = cv("enc_conv3", pool2); pool3 = maxpool2(x)
    x = cv("enc_conv4", pool3); pool4 = maxpool2(x)
    x = cv("enc_conv5a", pool4)
    x = cv("enc_conv5b", x)
    x = np.concatenate([up2(x), pool3], axis=1); x = cv("dec_conv4a", x); x = cv("dec_conv4b", x)
    x = np.concatenate([up2(x), pool2], axis=1); x = cv("dec_conv3a", x); x = cv("dec_conv3b", x)
    x = np.concatenate([up2(x), pool1], axis=1); x = cv("dec_conv2a", x); x = cv("dec_conv2b", x)
    x = np.concatenate([up2(x), inp], axis=1);   x = cv("dec_conv1a", x); x = cv("dec_conv1b", x)
    x = cv("dec_conv0", x)  # relu6 to match unet.ts default
    return x


def reference_large(weights, inp):
    def cv(name, x, act=True):
        y = conv2d_same(x, weights[f"{name}.weight"], weights[f"{name}.bias"])
        return relu6(y) if act else y

    x = cv("enc_conv1a", inp); x = cv("enc_conv1b", x); pool1 = maxpool2(x)
    x = cv("enc_conv2a", pool1); x = cv("enc_conv2b", x); pool2 = maxpool2(x)
    x = cv("enc_conv3a", pool2); x = cv("enc_conv3b", x); pool3 = maxpool2(x)
    x = cv("enc_conv4a", pool3); x = cv("enc_conv4b", x); pool4 = maxpool2(x)
    x = cv("enc_conv5a", pool4); x = cv("enc_conv5b", x)
    x = np.concatenate([up2(x), pool3], axis=1); x = cv("dec_conv4a", x); x = cv("dec_conv4b", x)
    x = np.concatenate([up2(x), pool2], axis=1); x = cv("dec_conv3a", x); x = cv("dec_conv3b", x)
    x = np.concatenate([up2(x), pool1], axis=1); x = cv("dec_conv2a", x); x = cv("dec_conv2b", x)
    x = np.concatenate([up2(x), inp], axis=1);   x = cv("dec_conv1a", x); x = cv("dec_conv1b", x)
    x = cv("dec_conv1c", x)
    return x


def main():
    tza = sys.argv[1] if len(sys.argv) > 1 else "../../packages/denoiser/tzas/rt_hdr_alb_nrm_small.tza"
    size = 64
    weights = parse_tza(open(tza, "rb").read())
    if "enc_conv1a.weight" in weights:
        ref_fn, in_ch = reference_large, weights["enc_conv1a.weight"].shape[1]
    else:
        ref_fn, in_ch = reference, weights["enc_conv0.weight"].shape[1]

    out_path = "/tmp/parity.onnx"
    convert(tza, out_path, size, size, fp16=False, final_activation="relu6")

    rng = np.random.default_rng(0)
    inp = rng.random((1, in_ch, size, size), dtype=np.float32)

    ref = ref_fn(weights, inp)
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(["output"], {"input": inp})[0]

    print("shapes ref/onnx:", ref.shape, got.shape)
    diff = np.abs(ref - got)
    print(f"max abs diff: {diff.max():.3e}   mean abs diff: {diff.mean():.3e}")
    ok = np.allclose(ref, got, atol=1e-4, rtol=1e-4)
    print("PARITY OK" if ok else "PARITY FAIL")

    # Dynamic-dims export: run a batch of 2 through [batch, C, height, width]
    # and check each item matches its independent reference (validates that
    # batching through conv/pool/resize/concat keeps items independent).
    dyn_path = "/tmp/parity_dynamic.onnx"
    convert(tza, dyn_path, "height", "width", fp16=False, final_activation="relu6")
    inp2 = rng.random((1, in_ch, size, size), dtype=np.float32)
    batch = np.concatenate([inp, inp2], axis=0)
    sess_d = ort.InferenceSession(dyn_path, providers=["CPUExecutionProvider"])
    got_b = sess_d.run(["output"], {"input": batch})[0]
    ref2 = ref_fn(weights, inp2)
    d0 = np.abs(ref - got_b[0:1]).max()
    d1 = np.abs(ref2 - got_b[1:2]).max()
    ok_dyn = d0 < 1e-4 and d1 < 1e-4
    print(f"dynamic batch=2 max abs diff: item0 {d0:.3e}  item1 {d1:.3e}")
    print("DYNAMIC PARITY OK" if ok_dyn else "DYNAMIC PARITY FAIL")
    sys.exit(0 if (ok and ok_dyn) else 1)


if __name__ == "__main__":
    main()
