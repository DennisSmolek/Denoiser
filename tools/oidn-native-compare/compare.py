"""Compare PFM images: PSNR + optional tonemapped PNG export for eyeballing.

Usage: python compare.py a.pfm b.pfm [--png]   # PSNR(a, b) + a.png/b.png
       python compare.py a.pfm --png           # just export a.png
"""
import sys

import numpy as np


def read_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
    img = data.reshape(h, w, 3)
    return np.flipud(img)  # -> row 0 = top


def to_png(path, img):
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("pip install pillow for --png")
    v = np.clip(img, 0, None)
    v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)  # ACES approx
    v = np.clip(v, 0, 1)
    v = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
    Image.fromarray((v * 255).astype(np.uint8)).save(path)
    print(f"wrote {path}")


args = [a for a in sys.argv[1:] if not a.startswith("--")]
png = "--png" in sys.argv

a = read_pfm(args[0])
if png:
    to_png(args[0].replace(".pfm", ".png"), a)
if len(args) > 1:
    b = read_pfm(args[1])
    if png:
        to_png(args[1].replace(".pfm", ".png"), b)
    mse = float(np.mean((a - b) ** 2))
    print(f"MSE {mse:.6e}  PSNR(1.0 ref) {10 * np.log10(1.0 / mse):.2f} dB")
