# ORT-web WebGPU vs WASM — OIDN aux (multi-channel) miscompute repro

**Question answered:** is the aux denoising speckle *our* bug or onnxruntime-web's?
**Answer: onnxruntime-web's WebGPU execution provider.** Not us.

This runs the OIDN U-Net ONNX models through **bare, vanilla ORT-web sessions**
(no denoiser-library code, no gpu-buffer IO, no `freeDimensionOverrides`, plain
CPU-array in/out) on the **same fixed synthetic input**, comparing the **WebGPU**
provider against the **WASM** (CPU) provider — which matches native OIDN and the
Python CPU provider. The only variable is the execution provider.

## Result (onnxruntime-web 1.27.0, headless Chrome, ANGLE/Metal, M-series)

| model | input channels | max\|Δ\| WebGPU-vs-WASM | verdict |
|---|---|---|---|
| `rt_hdr` (color) | **3** | 9.5e-7 | ✅ matches |
| `rt_hdr_alb` | **6** | 3.7e-2 | ❌ diverges |
| `rt_hdr_calb_cnrm` (aux base) | **9** | 1.1e-1 | ❌ diverges |
| `rt_hdr_calb_cnrm_small` (aux) | **9** | 1.5e-1 | ❌ diverges |
| `rt_hdr_calb_cnrm.fp16` (aux) | **9** | 1.1e-1 | ❌ diverges (fp16 too) |

**The divergence appears above 3 input channels and scales with the count**
(3 → clean, 6 → 0.037, 9 → 0.108). The WebGPU output is visibly noisier and
skews more negative (e.g. min −0.051 vs WASM −0.015). Every model has the same
U-Net topology and internal convs with 32–48 channels — those are fine in the
3-channel model — so the defect is specifically ORT-web's WebGPU **Conv**
handling of the **first layer's input-channel reduction** when the model input
has more than 3 channels. fp32 and fp16 both affected; base and small both
affected.

Our converted ONNX is correct: it matches native OIDN to 54 dB on the CPU/WASM
provider (see `../oidn-native-compare`). The port is not at fault.

## Isolation: it is the *graph-input* conv specifically, not any wide conv

A second test (`../ort-webgpu-aux-split`) narrows the fault precisely. The OIDN
U-Net reduces the raw 9-channel `input` tensor in **two** places:

- `enc_conv0` — the first layer (Conv `9 → 32`, 3×3), and
- `dec_conv1a` — a decoder conv fed by `concat_38 = concat(up_37, input)`, i.e.
  it reduces `96 + 9 = 105` input channels, **including the same raw 9ch input**.

We split the graph at `enc_conv0`, computed that one conv on the WASM/CPU
provider (the correct result), and ran **the entire rest of the network —
including `dec_conv1a` with its raw-input skip — on the WebGPU provider**. Result:

| pipeline | max\|Δ\| vs WASM reference | verdict |
|---|---|---|
| full model on WebGPU | 1.08e-1 | ❌ diverges (the bug) |
| head+tail both on WASM (split sanity) | 0.0 | ✅ split is faithful |
| **`enc_conv0` off-GPU + entire tail on WebGPU** | **1.19e-6** | ✅ **matches reference** |

So `dec_conv1a` — a *wider* conv (105 inputs) that also ingests the raw 9-channel
input — computes **correctly** on the WebGPU EP. The fault is not "any conv with
> 3 input channels" and not "any conv touching the raw input"; it is specifically
the **model's first Conv reducing the raw graph-input tensor when that input has
> 3 channels**. Removing just that one conv from the WebGPU path restores full
native-quality output (1.2e-6). This strongly suggests the bug is in how the
WebGPU EP prepares/packs the *graph input* for the initial Conv (layout, channel
tiling, or the input-channel reduction on that first op), not in the Conv kernel
in general.

## Run

```sh
python3 -m http.server 5179        # from this directory
# open http://localhost:5179 in a WebGPU browser; results print + land on
# window.__results
```

Fully self-contained: ORT loads from the jsDelivr npm CDN, models from the
`pmndrs/denoiser-weights` CDN. Nothing local to build. Portable enough to attach
to an onnxruntime issue as-is.

## Implications / next steps

- **File upstream** (onnxruntime): the WebGPU EP miscomputes the **first Conv
  that reduces the raw graph-input tensor when that input has > 3 channels**;
  magnitude scales with channel count; WASM/CPU are correct; other (wider) convs
  reducing the same raw input mid-graph are fine (see isolation above). Related
  open reports: microsoft/onnxruntime #24070, #26734, #24442.
- **fp16 is NOT a workaround** (still diverges).
- **GPU-side workaround — VERIFIED (not just hypothesised):** compute the one
  bad conv (`enc_conv0`) ourselves in WGSL (our conv is correct to 2e-6, see the
  `kernel-spike` branch), re-export the model to start at `enc_conv1` (two inputs:
  the `enc_conv0` feature map **and** the raw `input`, which the `dec_conv1a`
  skip still needs), and let ORT run the rest on WebGPU. Measured end-to-end:
  **1.2e-6 vs the native/WASM reference** — full quality, no CPU fallback. The
  standalone proof is in `../ort-webgpu-aux-split`.
- **Also worth trying:** a newer/nightly ORT-web build (may already be fixed).
- **Product stance meanwhile:** keep aux off by default (color-only, 3ch, is
  clean and fast); mark aux experimental.
