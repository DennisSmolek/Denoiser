# WebGPU + ONNX Migration — Progress & Handoff

> **Historical record.** The migration completed; current state and next
> actions live in [`STATUS.md`](STATUS.md). The TODO section at the bottom is
> superseded (2c/G-buffer aux, zero-copy IO, and the docs/example cleanup all
> landed on `perf-v2`). Kept for the architecture notes and gotchas.

Branch: **`feat-v2`**. This documents the migration of the `denoiser` library off
TensorFlow.js/WebGL onto a full-WebGPU stack (onnxruntime-web) plus the three.js
r185 WebGPU path tracer example. Written as a resumable checkpoint.

## Goals (both delivered & browser-verified)
1. **Replace TensorFlow.js** (abandoned) with **onnxruntime-web** (WebGPU EP now,
   WebNN later) for the OIDN U-Net. ✅ `packages/denoiser` is fully WebGPU/ONNX; TFJS removed.
2. **WebGPU path tracer**: three r185 + the unreleased `WebGPUPathTracer` feeding the
   denoiser on one shared GPUDevice. ✅ Working example.

## Status: what's DONE (all verified in a WebGPU browser)
- **ONNX conversion tooling** (`tools/onnx-convert/`, Python, no PyTorch): builds ONNX
  U-Nets directly from OIDN `.tza` weights, 1:1 with the old `unet.ts` (OIHW conv,
  relu6/Clip, 2×2 maxpool, nearest Resize upsample, channel-axis skip-concat). Auto-detects
  standard (32-tensor) vs UNetLarge (38-tensor; the old TF `_buildLarge` was a stub).
  NumPy-vs-ORT parity ~1e-6. Run: `convert.py tzas/*.tza -o packages/denoiser/models [--fp16]`.
  Verify: `verify_parity.py`. 23 variants × fp32+fp16 generated (gitignored, CDN-hosted).
- **Library cutover** (`packages/denoiser`): fully rewritten on WebGPU/ORT, TFJS dropped.
  Builds clean (`tsc && rollup`, 29KB, ORT external). Browser-verified ~28ms warm (tiled 384²).
- **Examples** (under `/examples`, gitignored — source force-added per file):
  - `webgpu-ort-smoke` — engine harness: ORT WebGPU EP, gpu-buffer IO, WGSL pre/post,
    tiling+blend, 9ch aux. (~5ms/256 tile warm, GPU-vs-JS diff 0.)
  - `denoiser-package-test` — the real `Denoiser` package end-to-end (~28ms).
  - `three-pathtracer-webgpu` — three r185 WebGPUPathTracer → denoiser, shared device.

## Library architecture (`packages/denoiser/src`)
- `denoiser.ts` — public `Denoiser` class (WebGPU-only). API kept close to old: `execute`,
  `setInputImage`/`setInputData`, `setCanvas`, `onExecute`/`onProgress`/`onBackendReady`,
  props (`quality`/`hdr`/`srgb`/`height`/`width`/aux flags), `weightsUrl`/`weightsPath`,
  `flipOutputY`, `build`, `dispose`. Exposes **`denoiser.device`** (ORT's GPUDevice) to share.
- `ort/engine.ts` — `DenoiseEngine`: ORT InferenceSession + IO-bound gpu-buffer tensors,
  256² tiling (overlap 32, stride 224) with overlap-blend, on the shared device.
- `ort/wgsl.ts` — `GpuImageOps`: WGSL compute kernels — extract/normalize/HWC→NCHW
  (3/6/9ch concat + sRGB→linear), accumulate (min-of-sigmoid blend), resolve (→RGBA, sRGB, clamp).
- `weights.ts` — `Models`: fetch+cache `.onnx` by name (replaces TZA parser). CDN-hosted.
- `modelName.ts` — props → ONNX model name (port of `determineTensorMap`) + channel count.
- `utils.ts` — pure-DOM helpers (image→RGBA, flip, formatTime). `types.ts`, `global.d.ts`.
- **Deleted**: `tza.ts`, `unet.ts`, `tiler.ts`, `denoiserUtils.ts`, `webglStateManager.ts`.
- **Deps**: dropped `@tensorflow/*`; added `onnxruntime-web` (external in rollup), `@webgpu/types`.

## Key facts / gotchas (don't re-discover these)
- **OIDN weights already current**: our 21 RT `.tza` are byte-identical to upstream
  `oidn-weights` master. OIDN 2.x gains are engine-side, not retrained weights. Added the 2
  lightmap models. Weight refresh = drop new `.tza` + rerun converter.
- **relu6 on the OUTPUT conv** matches the old deployed TF behavior (converter default);
  `--final-activation none` is closer to upstream OIDN but changes results. User's call.
- **Shared device (#26107)**: ORT creates the GPUDevice and ignores an injected one. So ORT
  must init first; three.js borrows `ort.env.webgpu.device`. In the pathtracer example we
  **monkey-patch `requestDevice` for max limits/features** before any device creation, or the
  path tracer's megakernel compute pipeline fails validation on ORT's minimal device.
- **WebGPUPathTracer (unreleased branch `gkjohnson/three-gpu-pathtracer#webgpu-pathtracer`)**:
  - `./webgpu` export = ESM source (no build step). Peer: three ≥0.180, three-mesh-bvh ≥0.9.9, xatlas-web.
  - **Must call `useMegakernel(true)`** — the constructor's default tracer has no material wired in
    (else `bsdfSample of null`).
  - **Analytic lights unsupported** — light via environment. Use `GradientEquirectTexture`
    (from `three-gpu-pathtracer/src/textures/...`); a raw `DataTexture` trips three r185's compute
    sampler codegen (`nodeUniformN_sampler` unresolved).
  - Output is a **linear-HDR float `StorageTexture`** at `pathTracer._pathTracer.outputTarget`
    (public `.target` doesn't exist). Read via `{ textures: [target] }` +
    `readRenderTargetPixelsAsync`, then tonemap (ACES) + sRGB. `drawImage` on a WebGPU canvas → black.
  - WebGPU render targets read **bottom-up** → `flipOutputY`.
- **Vite for the pathtracer example**: `optimizeDeps.exclude` three/three subpaths/three-mesh-bvh/
  three-gpu-pathtracer + `resolve.dedupe:['three']` (single instance, or sampler codegen breaks);
  `esnext` target (pathtracer uses top-level await). Models served at `/models` via a dev middleware.
- **Old examples** (`three-pathtracer`, `three-pathtracer-texture-in`, `threejs-webgl`,
  `webgl-basic`, `basic`, `webgpu-basic`, `upload`) use the OLD TFJS API + pull `three@0.166` /
  `three-mesh-bvh@0.7.6` into root node_modules (the dual-three foot-gun). To be deleted.

## Commits (on feat-v2, newest first)
- `f3f9ae9` Phase 2: working three r185 WebGPUPathTracer → denoiser pipeline
- `f8f8602` Phase 2a: pathtracer example + dependency validation
- `a3d47ed` denoiser package end-to-end test example
- `e6d1941` **library rewrite — WebGPU/ONNX, TFJS removed**
- `d5ae938` WGSL/ORT smoke example (engine harness)
- `3ae4231` TZA→ONNX conversion tooling
- `e34a0b3` lightmap weights

## How to run / verify
```sh
# 1) Convert models (once)
cd tools/onnx-convert && python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python convert.py ../../packages/denoiser/tzas/*.tza -o ../../packages/denoiser/models        # + --fp16

# 2) Build the library
corepack yarn install
cd packages/denoiser && yarn build          # tsc + rollup

# 3) Run an example in a WebGPU browser (dev mode serves /models)
cd examples/three-pathtracer-webgpu && yarn dev      # or webgpu-ort-smoke / denoiser-package-test
```

## Perf work — DONE (branch `perf-v2`, July 2026)
**See `perf-plan.md` (plan + measured results).** Executed phases: bench harness
(`examples/bench`), graph capture (opt-in only — ORT 1.27 crash + no gain), dynamic-dim
models + batched tiles, true fp16 (fixed: was unusable), whole-frame runs + geometry
ladder + scoped max-limits device patch, zero-copy texture IO (**2b DONE** — pathtracer
example has a zero-copy HDR button; 512² 68.8→35.4ms). 1080p CPU-input path:
188→104ms (fp16). PSNR fp16 vs fp32: 53.5dB. Critical fix: `build()` overlaps engine
creation/disposal so releasing the last ORT session can't destroy the shared device.
Models re-exported with named dynamic dims (`--size` for legacy static).

## TODO (remaining)
- **2c — TSL MRT G-buffer**: render albedo + view-normal (`mrt({ output, diffuse, normal })`)
  → the 9-channel aux model, validating aux on real render data (setInputTexture supports it).
- **Cleanup**: delete the old WebGL/TFJS examples (also removes dual three@0.166); update
  `packages/denoiser/README.md` + root `README.md` (still reference tensorflow.js); changeset/major bump.
- **WebNN track**: deferred pending user research — plan in `perf-plan.md`.
- **Later phase (user request)**: investigate where OIDN 2.x's perf gains came from (engine/data
  handling, since weights are identical) — see memory `oidn-weights-already-current`.
