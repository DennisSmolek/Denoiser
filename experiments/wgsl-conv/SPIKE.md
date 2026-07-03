# WGSL conv kernel spike (branch: kernel-spike)

Goal (docs/wgsl-engine-proposal.md phase 1): can custom WGSL conv kernels beat
ORT-web's by ≥1.4× on the shapes that dominate the U-Net? Measured ceiling:
native Metal runs the base net 9× faster than ORT-web (matrix units + fusion).

## Design principles (per Dennis)

- **Kernel registry**: every idea is a variant in `kernels.js` — name + WGSL +
  dispatch. The harness runs ALL of them; each gets a correctness gate
  (max-abs-diff vs the ORT reference output on identical random weights/input)
  and a timing. A wrong or slow kernel is a data point, not a breakage.
- **Nothing here touches packages/denoiser.** Promotion to the library only
  happens after a variant wins on correctness + speed across sizes.
- Native-side tooling (quantization calibration etc.) is fair game — same flow
  as the ONNX conversion: do the hard math offline in python, ship artifacts.

## Layout

- `make_conv_model.py` — emits `conv.onnx` (3×3 Conv, Cin→Cout, random weights,
  dynamic H/W) + `weights.bin`/`bias.bin` so ORT and our kernels use IDENTICAL
  parameters.
- `main.js` — browser harness (no build step; ORT from CDN): runs ORT-WebGPU as
  reference+baseline, then every registry kernel: correctness, then N timed
  iterations bracketed by onSubmittedWorkDone.
- `kernels.js` — the registry. Seed variants:
  - `naive` — 1 thread per (x, y, cout), scalar loop. Correctness anchor.
  - (next) `tiled-smem` — 16×16 workgroup, input tile+halo in workgroup memory,
    channel-chunked (18×18×8ch ≈ 10KB fits the 32KB limit), unrolled 3×3.
  - (next) `tiled-f16` — same with f16 storage/arithmetic (2× bandwidth).
  - (next) `fused-relu` / `fused-pool` — fold the activation/pool into the conv.
- Shapes: 512² and 1920×1088 at Cin=Cout=64 (the base net's dominant full-res
  decoder shape). Report a table.

## Run

```sh
python3 make_conv_model.py          # uses tools/onnx-convert's venv deps
python3 -m http.server 5178         # then open http://localhost:5178
```

## Status

- [x] scaffold + naive kernel + harness
- [x] verified in browser: naive matches ORT to 2e-6 at 512p and 1080p
- [ ] FIX BASELINE: ORT ran with CPU-tensor IO (67MB copies/run) — rerun with
  gpu-buffer IO before believing any "Nx vs ORT" number (naive "3.75x/4.59x" is
  inflated by ORT paying transfers our kernels do not)
- [ ] tiled-smem variant
- [ ] f16 variant
- [ ] fusion experiments
- [ ] go/no-go: ≥1.4× over ORT on both sizes?
