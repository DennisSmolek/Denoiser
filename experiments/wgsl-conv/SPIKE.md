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
- [x] FIX BASELINE: ORT now runs gpu-buffer IO (`Tensor.fromGpuBuffer` +
  `preferredOutputLocation:'gpu-buffer'`) — no CPU copies in the timed loop.
- [x] tiled-smem variants (3: 8/16/32 output-channels-per-thread)
- [x] f16 variants (3: base, co32, acc32)
- [x] fusion experiments (relu6 folded into the f32 and f16 epilogues)
- [x] go/no-go — **NO-GO** (see verdict). Best custom kernel ~1.2–1.32× at 512²,
  under the ≥1.4× gate.

## Results

Conv 64→64, 3×3, the base net's dominant full-res decoder shape. M-series Mac,
headless Chrome (ANGLE/Metal, `shader-f16` present). warm 3, 12 timed iters,
bracketed by `onSubmittedWorkDone`. Correctness = max-abs-diff vs the ORT
reference on identical weights/input (f32 gate 1e-3, f16 gate 1e-1).

**512×512** — ORT-WebGPU gpu-buffer baseline ≈ **5.5–6.7 ms** (±~20% run to
run; speedups below use the same-run baseline):

| kernel | ms | ×ORT | maxdiff | notes |
|---|---|---|---|---|
| naive (anchor) | ~37 | 0.18× | 2e-6 | correctness anchor, not a contender |
| tiled-smem (8 co/thr, 8ch) | ~8.5 | 0.79× | 2e-6 | f32 smem, slower than ORT |
| tiled-smem-co16 | ~7.1 | 0.95× | 2e-6 | |
| tiled-smem-co32 (4ch) | ~6.3 | 1.08× | 2e-6 | best f32, barely beats ORT |
| tiled-f16 (16 co/thr, 16ch) | ~5.6 | 1.20× | 1.4e-2 | |
| tiled-f16-co32 (8ch) | ~5.1 | **1.31×** | 1.4e-2 | best pure-conv |
| tiled-f16-acc32 | ~5.6 | 1.21× | 1.2e-3 | f16 dot, f32 accumulate (tighter) |
| fused-relu6 (f32) | ~7.0 | 0.97× | 2e-6 | |
| fused-relu6-f16 | ~5.1 | **1.32×** | 1.4e-2 | best overall (fusion is ~free here) |

**1920×1088** — **not measurable in this headless setup.** The harness needs
several ~534 MB buffers live at once (f32 in + f32 out + f16 in/weights/out ≈
2 GB+), and the ORT baseline `session.run` hangs the headless GPU before
producing a number across repeated attempts. A real card / non-headless run
should get it, but 512² already decides the gate.

## Verdict: NO-GO (do not build the full bespoke engine on this evidence)

- **Best custom kernel ≈ 1.3× over a fair ORT-WebGPU baseline, under the 1.4×
  gate** — and that best case is f16-only (1.4e-2 error) with fusion; every
  **f32** tiled kernel lands ≤1.08×, i.e. roughly at parity with ORT.
- The baseline itself is noisy (±20%), so "1.3×" is soft. There's no dominant
  win here — ORT's WebGPU conv is already close to what straightforward tiled
  WGSL achieves.
- This matches the native-comparison conclusion (`../../docs/status/speedup.md`,
  `../../tools/oidn-native-compare`): the ~9× gap to native Metal is **hardware
  matrix units** (Apple `simdgroup_matrix` via MPSGraph), not orchestration
  overhead. WGSL can't reach those until the WebGPU subgroup-matrix proposal
  ships; until then a hand-rolled engine buys ~1.3×, not the 2–3× that would
  justify replacing ORT.

### What would change the verdict / try next
- **WebGPU subgroup-matrix** (cooperative-matrix) when it lands — that's the
  path to the matrix units; re-run this spike then.
- **int8/DP4a** (`packed_4x8_integer_dot_product`) — 4 MACs/instr + 2× BW; needs
  offline weight+activation quantization with PSNR A/B (proposal phase 3).
- Measure 1080p on a real GPU (non-headless) to confirm the ratio holds at the
  bandwidth-bound size — f16 may widen slightly there, but unlikely past 1.4×.
