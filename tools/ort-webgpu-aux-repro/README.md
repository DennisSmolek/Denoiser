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

- **File upstream** (onnxruntime): Conv on the WebGPU EP miscomputes when the
  model input has >3 channels; magnitude scales with channel count; WASM/CPU are
  correct. Related open reports: microsoft/onnxruntime #24070, #26734, #24442.
- **fp16 is NOT a workaround** (still diverges).
- **Most promising GPU-side workaround:** the bug is isolated to the first conv
  (`enc_conv0`, >3→N). Our own WGSL conv is correct to 2e-6 (see the
  `kernel-spike` branch). So: compute `enc_conv0` in our kernel, re-export the
  model to start at layer 1, and let ORT run the (correct) rest on WebGPU. No
  CPU fallback needed.
- **Also worth trying:** a newer/nightly ORT-web build (may already be fixed).
- **Product stance meanwhile:** keep aux off by default (color-only, 3ch, is
  clean and fast); mark aux experimental.
