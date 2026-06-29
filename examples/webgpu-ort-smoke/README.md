# WebGPU ORT smoke test

Minimal end-to-end check of the migration's core: run a converted OIDN U-Net
(`rt_ldr_small.onnx`) through **onnxruntime-web's WebGPU execution provider**
with **zero-copy GPU-buffer IO** (`ort.Tensor.fromGpuBuffer` +
`preferredOutputLocation: 'gpu-buffer'`). No TensorFlow.js, no WebGL.

It also surfaces the GPUDevice ORT creates (`ort.env.webgpu.device`) — the same
device we'll hand to three.js's `WebGPURenderer` later to keep everything on one
device (see onnxruntime issue #26107).

## Prereqs

1. Convert the models first (writes to `packages/denoiser/models/`):
   ```sh
   cd ../../tools/onnx-convert
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   python convert.py ../../packages/denoiser/tzas/rt_ldr_small.tza -o ../../packages/denoiser/models
   ```
2. A browser with WebGPU (Chrome/Edge ≥ 121, or Safari/Firefox with WebGPU on).

## Run

```sh
npm install      # or: yarn
npm run dev      # vite dev server; open the printed URL
```

You should see a noisy synthetic image on the left and a visibly smoothed
(denoised) image on the right, plus timing in the status box. "Re-run" makes a
fresh noisy image and denoises again.

## Two paths in the demo

- **`denoiseTileGPU`** (default, what's displayed + benchmarked) — the full-GPU
  path: RGBA8 → normalized NCHW via a **WGSL compute** kernel, inference with
  **IO-bound** input/output GPU tensors (reused buffers, no per-run alloc),
  NCHW → RGBA8 via WGSL, single readback for the canvas. This is the shape the
  real library will take.
- **`denoiseTile`** (JS) — does layout/normalize on the CPU; kept only as an A/B
  reference. On load, the page logs the **max pixel diff between the two paths**
  (expect 0–1, pure rounding) to prove the WGSL pre/post is correct.

There's also a **full-image tiled path** (`denoiseImage` + the "Denoise full image"
button): a 640×384 noisy image is split into 256² tiles (overlap 32, stride 224),
each tile run through the model, and the results blended with a min-of-sigmoid mask
— the GPU/WGSL equivalent of the old `tiler.ts`. Edge tiles are zero-padded; only
the final image is read back. Check the result has **no visible seams** between tiles.

And an **aux-input path** (`denoiseAuxTile` + "Denoise with aux") using the 9-channel
`rt_ldr_alb_nrm_small` model: a WGSL kernel concatenates color + albedo + normal into
a 9-channel NCHW input (color/albedo `[0,1]`, normal `[-1,1]`). The demo feeds a noisy
scene as color, the clean scene as albedo, and a flat synthetic normal — synthetic, so
it's a plumbing check (the model runs with concatenated aux), not a quality benchmark.

## What it proves / what it doesn't

- ✅ Exported ONNX graph runs on the WebGPU EP, produces a sane denoised image.
- ✅ Normalization + HWC↔NCHW layout done in WGSL compute on the shared device.
- ✅ IO-bound GPU tensors — input/output stay on the GPU, only final pixels read back.
- ✅ Tiling + overlap-blend of arbitrary-size images in WGSL (replaces tiler.ts).
- ✅ 9-channel aux concat (color+albedo+normal) in WGSL → multi-input models run.
- ⛔ sRGB↔linear and HDR autoexposure — trivial WGSL, baked in during the library fold-in
  and validated against real pathtracer output in Phase 2.

The engine's hard parts are now proven; next is folding it into `packages/denoiser`.
