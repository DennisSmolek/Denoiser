# OIDN TZA → ONNX converter

Build-time tooling that converts OIDN's `.tza` weight blobs (in
[`packages/denoiser/tzas/`](../../packages/denoiser/tzas/)) into ONNX U-Net
models for runtime inference with `onnxruntime-web` (WebGPU EP).

This replaces the old approach where the U-Net graph was rebuilt at runtime from
TZA weights with TensorFlow.js (`packages/denoiser/src/unet.ts`). ONNX Runtime
consumes a pre-serialized graph, so conversion happens here, offline, once.

## What it does

- Parses the TZA blob ([`tza.py`](tza.py), a port of the old `tza.ts` parser).
  Conv weights are OIHW — already the layout ONNX `Conv` wants, so no transpose.
- Builds the ONNX graph directly from the weights ([`convert.py`](convert.py)),
  mirroring `unet.ts` 1:1: 3×3 same-pad conv + bias, **relu6** (`Clip[0,6]`)
  activation, 2×2/stride-2 max pool, nearest 2× upsample (`Resize`,
  asymmetric/floor to match TF `upSampling2d`), channel-axis skip-concat.
- Auto-detects topology from the weight names: standard U-Net (`enc_conv0…`,
  32 tensors) or **UNetLarge** (`enc_conv1a/1b…`, two convs per stage, 38
  tensors). Large was never implemented in the TF version — ONNX gets it free.
- Emits **NCHW, fixed-shape** models (default 256×256) so the WebGPU EP can use
  graph capture. Channel count is inferred per variant (3 / 6 / 9 input ch).

## Usage

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# convert every variant into packages/denoiser/models/
python convert.py ../../packages/denoiser/tzas/*.tza -o ../../packages/denoiser/models

# fp16 (≈half size/bandwidth; needs the shader-f16 feature at runtime)
python convert.py ../../packages/denoiser/tzas/*.tza -o ../../packages/denoiser/models --fp16

# options: --size 256  --fp16  --final-activation relu6|none
```

`--final-activation relu6` (default) matches the current `unet.ts`, which applies
relu6 to the output conv too. Use `none` for an unclamped final conv (closer to
upstream OIDN); validate output quality if you change it.

## Verifying

[`verify_parity.py`](verify_parity.py) builds a NumPy reference of the same graph
and compares it against onnxruntime output (small input, both topologies). This
guards the export against OIHW / pad / pool / upsample / concat-axis mistakes.

```sh
python verify_parity.py ../../packages/denoiser/tzas/rt_hdr_alb_nrm_small.tza   # std
python verify_parity.py ../../packages/denoiser/tzas/rt_alb_large.tza           # large
# -> "PARITY OK", max abs diff ~1e-6
```

## Notes

- Models are not committed to npm (kept out like the `tzas/`); host on a CDN and
  point the library's `weightsUrl`/model path at them.
- Pre/post-processing (normalize, sRGB↔linear, color+albedo+normal concat) is
  intended to be baked into the graph and/or done in WGSL compute in the library;
  this converter currently emits the bare U-Net (already-normalized NCHW input).
