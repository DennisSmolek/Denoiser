# Proposal: bespoke WGSL inference engine for the OIDN U-Net (side project)

> **ARCHIVED — measured NO-GO (2026-07-07).** Best custom kernel hit 1.32×
> (f16), under the ≥1.4× gate; the native-Metal gap is hardware matrix units,
> unreachable from WGSL today. Results: `kernel-spike` branch,
> `experiments/wgsl-conv/SPIKE.md`. Revisit when WebGPU subgroup-matrix ships.

## Why

After the perf work, 1080p ≈ 104ms is ~100% ORT conv kernel time. ORT's WebGPU
kernels are general-purpose: per-op dispatches, intermediate tensors written to
and re-read from VRAM between every layer, generic layouts. Our workload is ONE
fixed topology (Conv3x3+relu6, MaxPool2, nearest-up, concat) with weights known
at load time — the textbook case where a specialized engine wins.

## Where the headroom is

- **Layer fusion.** conv→relu6 (free), conv→maxpool (halves one full-res
  write+read), up→concat→conv (the decoder reads pool skip + upsampled input
  directly instead of materializing the concat — saves a full-res tensor
  round-trip per decoder level). The U-Net's traffic is dominated by full-res
  intermediates; fusion cuts VRAM bandwidth ~30-45%.
- **Tiled shared-memory convolution.** Load an input tile + halo into workgroup
  storage once, compute all output channels from registers; f16 arithmetic with
  `shader-f16`; subgroup ops where available. ORT does some of this generically;
  a fixed-shape kernel tunes tile/channel blocking exactly.
- **Weight baking.** Weights preloaded once into a single packed buffer,
  layouts chosen for the kernels (e.g. OIHW→HWIO swizzled per tile plan), no
  per-run graph walking at all.
- **Zero runtime overhead.** No WASM boundary, no session machinery; the whole
  U-Net becomes ~30 precompiled pipelines dispatched from one command encoder.

Reference point: our WGSL pre/post kernels cost ~0.1ms while ORT runs ~100ms of
convs at 1080p — even 1.5× conv efficiency dwarfs everything else available.
**Estimated win: 1.5–2.5× vs ORT** (fusion + fp16 tiles), putting 1080p in the
40–70ms range on the same hardware.

## int8 (asked): worth it?

WGSL has `dot4I8Packed` (DP4a) behind the `packed_4x8_integer_dot_product`
feature — 4 int8 MACs per instruction, plus 2× bandwidth vs fp16. In the best
case that's ~1.5–2× over fp16 conv. BUT: (a) feature availability is uneven
across adapters, (b) it requires quantizing OIDN's weights + activations
(calibration, per-channel scales) — real quality risk for a *denoiser* whose
output is judged perceptually, (c) ORT-web won't do it for us (int8 conv on the
WebGPU EP is effectively absent). Verdict: **not as an ORT option today; viable
as phase 3 of THIS engine** where we control quantization and can A/B PSNR.
fp16 first — it's free quality-wise and we already ship fp16 weights.

## Plan (side project, independent of the main library)

1. **Spike (1-2 weeks):** single fused conv3x3+relu6 tiled kernel, benchmarked
   against ORT's conv on the same shapes (256², 1080p; fp32/fp16). Go/no-go
   gate: ≥1.4× on the dominant shapes.
2. **Full graph:** codegen the ~30 pipelines from the TZA weights directly
   (reuse tools/onnx-convert's parser; ONNX no longer needed in this path),
   ping-pong buffer plan, decoder fusion.
3. **Parity + bench:** PSNR vs ORT output (target ≥50dB), bench page A/B.
4. **Integration:** `DenoiseEngine` grows a `backend: 'ort' | 'wgsl'` switch —
   same Denoiser API, ORT stays the fallback/reference.
5. **Phase 3 (optional):** int8/DP4a path with per-channel quantization.

Risks: kernel tuning is GPU-specific (need M-series + NVIDIA + Intel checks);
maintenance of hand kernels; ORT improving under us (re-bench each release).
