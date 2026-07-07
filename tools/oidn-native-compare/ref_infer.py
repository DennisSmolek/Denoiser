"""Reference runner: our converted ONNX aux model + OIDN's exact color pipeline,
in Python/ORT (CPU). Isolates the MODEL+preprocessing (this script) from our
WGSL runtime engine — if this output is clean but the browser's ours_aux.pfm
speckles, the bug is in the engine, not the model.

Pipeline per docs/specs/oidn-color-reference.md.
Usage: python ref_infer.py [model.onnx]  (default rt_hdr_calb_cnrm.onnx)
"""
import sys
import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "../../packages/denoiser/models/rt_hdr_calb_cnrm.onnx"

# --- PU transfer (HDR) ---
a, b, c, d = 1.41283765e+03, 1.64593172e+00, 4.31384981e-01, -2.94139609e-03
e, f, g = 1.92653254e-01, 6.26026094e-03, 9.98620152e-01
y0, y1 = 1.57945760e-06, 3.22087631e-02
HDR_Y_MAX = 65504.0


def pu_forward(y):
    y = np.asarray(y, np.float32)
    out = np.where(y <= y0, a * y,
                   np.where(y <= y1, b * np.power(y, c) + d, e * np.log(y + f) + g))
    return out.astype(np.float32)


def pu_inverse(x):
    x0, x1 = 2.23151711e-03, 3.70974749e-01
    x = np.asarray(x, np.float32)
    out = np.where(x <= x0, x / a,
                   np.where(x <= x1, np.power((x - d) / b, 1.0 / c), np.exp((x - g) / e) - f))
    return out.astype(np.float32)


xMax = float(pu_forward(HDR_Y_MAX))
normScale = 1.0 / xMax


def read_pfm(p):
    with open(p, "rb") as fh:
        assert fh.readline().strip() == b"PF"
        w, h = map(int, fh.readline().split())
        s = float(fh.readline())
        data = np.fromfile(fh, dtype="<f4" if s < 0 else ">f4")
    return np.flipud(data.reshape(h, w, 3)).astype(np.float32)  # row 0 = top


def write_pfm(p, img):
    h, w, _ = img.shape
    with open(p, "wb") as fh:
        fh.write(b"PF\n")
        fh.write(f"{w} {h}\n".encode())
        fh.write(b"-1.0\n")
        np.flipud(img).astype("<f4").tofile(fh)


def autoexposure(color):
    """OIDN HDR inputScale (docs §Autoexposure)."""
    key, eps = 0.18, 1e-8
    H, W, _ = color.shape
    c = np.nan_to_num(color, nan=0.0)
    c = np.clip(c, 0, np.finfo(np.float32).max)
    L = 0.212671 * c[..., 0] + 0.715160 * c[..., 1] + 0.072169 * c[..., 2]
    nbh, nbw = -(-H // 16), -(-W // 16)  # ceil
    sums, cnt = 0.0, 0
    for by in range(nbh):
        y0i, y1i = by * H // nbh, (by + 1) * H // nbh
        for bx in range(nbw):
            x0i, x1i = bx * W // nbw, (bx + 1) * W // nbw
            meanL = L[y0i:y1i, x0i:x1i].mean()
            if meanL > eps:
                sums += np.log2(meanL)
                cnt += 1
    return float(key / np.exp2(sums / cnt)) if cnt else 1.0


color = read_pfm("color.pfm")
albedo = np.clip(np.nan_to_num(read_pfm("albedo.pfm"), nan=0.0), 0, 1)
normal = np.clip(np.nan_to_num(read_pfm("normal.pfm"), nan=0.0), -1, 1) * 0.5 + 0.5

inputScale = autoexposure(color)
print(f"inputScale (autoexposure) = {inputScale:.5f}")

# input color: *inputScale -> clamp -> PU forward *normScale
cin = np.clip(np.nan_to_num(color * inputScale, nan=0.0), 0, np.finfo(np.float32).max)
cin = pu_forward(cin) * normScale

H, W, _ = color.shape
nchw = np.concatenate([cin, albedo, normal], axis=2).transpose(2, 0, 1)[None].astype(np.float32)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
out = sess.run(["output"], {"input": nchw})[0][0].transpose(1, 2, 0)  # HWC

# output color: clamp -> PU inverse (undo normScale) -> *outputScale
out = np.clip(np.nan_to_num(out, nan=0.0), 0, np.finfo(np.float32).max)
out = pu_inverse(out * xMax) / inputScale

write_pfm("ref_aux.pfm", out.astype(np.float32))
print("wrote ref_aux.pfm")
