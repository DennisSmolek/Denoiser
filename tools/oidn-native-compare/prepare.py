"""Convert the browser dumps (examples/three-pathtracer-webgpu/dumps) into PFM
files for the native oidnDenoise CLI.

Dump filename convention: <name>.<gpuFormat>.<size>  (raw RGBA rows, no padding)
  color  — the path tracer's linear-HDR output (BOTTOM-UP rows)
  albedo — raster G-buffer base color, [0,1]      (TOP-DOWN rows)
  normal — raster G-buffer view normals, [-1,1]   (TOP-DOWN rows)

PFM is written bottom-up (negative scale = little-endian), so the tracer's rows
pass through as-is and the G-buffer rows get flipped.

Usage: python prepare.py [dumps_dir] [out_dir]
"""
import glob
import os
import sys

import numpy as np

dumps = sys.argv[1] if len(sys.argv) > 1 else "../../examples/three-pathtracer-webgpu/dumps"
outdir = sys.argv[2] if len(sys.argv) > 2 else "."


def load(name):
    matches = glob.glob(os.path.join(dumps, f"{name}.*"))
    if not matches:
        raise SystemExit(f"missing dump: {name} (run window.__dumpForOIDN() in the demo)")
    path = matches[0]
    _, fmt, size = os.path.basename(path).split(".")
    size = int(size)
    dtype = np.float32 if "32float" in fmt else np.float16
    data = np.fromfile(path, dtype=dtype).reshape(size, size, 4).astype(np.float32)
    return data[:, :, :3]  # drop alpha


def write_pfm(path, img_topdown):
    """img_topdown: (H, W, 3) float32, row 0 = top. PFM stores rows bottom-up."""
    h, w, _ = img_topdown.shape
    with open(path, "wb") as f:
        f.write(b"PF\n")
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")  # negative = little-endian
        np.flipud(img_topdown).astype("<f4").tofile(f)
    print(f"wrote {path}")


color = np.flipud(load("color"))  # bottom-up -> top-down
albedo = load("albedo")           # already top-down
normal = load("normal")

write_pfm(os.path.join(outdir, "color.pfm"), color)
write_pfm(os.path.join(outdir, "albedo.pfm"), np.clip(albedo, 0, 1))
write_pfm(os.path.join(outdir, "normal.pfm"), np.clip(normal, -1, 1))
