# Native OIDN reference comparison

Answers "is our aux denoising wrong, or is this scene just a weak case for aux?"
by running the EXACT inputs our denoiser sees through Intel's native OIDN
(same weights, reference CPU implementation).

## 1. Get the native binary (one-time, needs your OK — it's a prebuilt from
Intel's official release page)

```sh
mkdir -p /tmp/oidn-native && cd /tmp/oidn-native
curl -sL https://github.com/RenderKit/oidn/releases/download/v2.5.0/oidn-2.5.0.arm64.macos.tar.gz | tar xz
OIDN=/tmp/oidn-native/oidn-2.5.0.arm64.macos/bin/oidnDenoise
```

## 2. Capture inputs from the demo

Run the pathtracer example, let it accumulate to the sample count you want to
test (e.g. 4), then in the browser console:

```js
await __dumpForOIDN()
```

Raw float dumps land in `examples/three-pathtracer-webgpu/dumps/`.

## 3. Convert + run native (this dir; needs numpy, pillow for PNGs)

```sh
python prepare.py                       # -> color.pfm albedo.pfm normal.pfm

$OIDN --hdr color.pfm -o native_color.pfm                                   # color-only
$OIDN --hdr color.pfm --alb albedo.pfm --nrm normal.pfm --clean_aux -o native_aux.pfm

python compare.py native_color.pfm native_aux.pfm --png   # PSNR + PNGs to eyeball
python compare.py color.pfm --png                          # the noisy input, for reference
```

## Interpreting

- **native_aux visibly better than native_color** (expected on most scenes) →
  our aux path has a remaining bug; diff our output against native_aux next.
- **native_aux ≈ native_color, both smooth** → our aux *pipeline* is fine and
  the earlier speckle came from somewhere else; compare ours vs native_color.
- **native_aux also speckles** → the scene/aux combination is genuinely hard
  (env-lit flat-albedo scene gives aux little signal); our port is not at fault.

Notes: `--clean_aux` matches our rasterized (noise-free) G-buffer and the
`calb_cnrm` model our library selects. Native applies its own autoexposure +
PU transfer internally — same as our implementation (docs/oidn-color-reference.md).

## Results (July 2026, M-series Mac, this repo's demo scene @ 4 samples)

**Quality:** `native_aux.png` is perfectly smooth on our exact dumped inputs —
so the web aux path's residual speckle is OUR bug, not a weak-aux scene.
PSNR(native_color, native_aux) = 47.1 dB (both clean; aux slightly sharper edges).

**Speed (oidnBenchmark, 1080p, 9ch hdr_alb_nrm = BASE model):**
| | ms/image |
|---|---|
| native Metal (GPU) | **24.5** |
| native CPU | 334.5 |
| web (ours), base model fp16 | ~224 |
| web (ours), small model fp16 | ~104 |

Takeaways: we beat native CPU already; native Metal proves this hardware runs
the base 9ch U-Net in ~25ms → ~9x kernel-efficiency headroom over ORT-web at
equal model size — the measured ceiling for docs/wgsl-engine-proposal.md.

### Fairness check (`-t half|float`, `-q high|balanced|fast`, Metal)

| network | native Metal | ours (WebGPU/ORT fp16) | gap |
|---|---|---|---|
| base (high/balanced) | 24.2 ms | ~224 ms | ~9x |
| fast (small) | 14.0 ms | ~104 ms | ~7.4x |

`-t half` vs `-t float` is IDENTICAL on Metal (24.2 vs 24.3) — that flag sets
buffer IO type; the MPSGraph backend runs its own (fp16-class, matrix-unit)
precision internally either way. So the native run is the optimized ceiling,
not a generic path. Native msec excludes host IO (~0.8ms, reported separately);
our numbers include extract/resolve/readback.

Gap decomposition (measured, not vibes): browser overhead is NOT the story —
after batching/whole-frame our dispatch cost is ~1-3ms; the 224ms is kernel
execution. Metal's edge = MPSGraph fused kernels + Apple simdgroup_matrix
(hardware MMA), the latter unreachable from WGSL until WebGPU's subgroup-matrix
proposal ships. Realistic custom-WGSL capture: ~2-3x (base -> ~80-120ms);
the remaining ~3-4x is hardware access, not code quality.
