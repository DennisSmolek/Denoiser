# Migrating from v1 (0.x, TensorFlow.js) to v2

v2 is a ground-up rewrite: TensorFlow.js and every WebGL path are gone; the
U-Net runs as ONNX on onnxruntime-web's WebGPU execution provider, and all
pre/post-processing is WGSL compute on one shared `GPUDevice`. It is a **clean
break** — the API changed deliberately (see
[`docs/archive/api-v2-spec.md`](../archive/api-v2-spec.md) for the rationale):
v1 configured runs through instance flags set in the right order; v2 is
**stateless per call** — the instance holds identity only (models, sessions,
device), every run is fully described by its arguments.

## Requirements changed

| | v1 | v2 |
|---|---|---|
| runtime | TensorFlow.js (WebGL or WebGPU backend) | onnxruntime-web ≥ 1.27 (peer dep), WebGPU only |
| browser | WebGL2 fallback existed | **WebGPU required** (no fallback) |
| weights | `.tza` files (`/tzas` CDN dir) | converted `.onnx` models (`/models`), fp32 + fp16 |
| install | `npm i denoiser` | `npm i denoiser onnxruntime-web` |

If you must support non-WebGPU browsers, stay on v1 (0.0.11) for those users —
v2 throws `DenoiserUnsupportedError` where WebGPU/features are missing.

## API mapping

### Construction

```ts
// v1 — sync constructor, backend + context injection, ready-listener
const denoiser = new Denoiser('webgpu', canvasOrDevice);
denoiser.onBackendReady(() => { /* ... */ });

// v2 — async factory; ORT creates the device, you share it OUTWARD
const denoiser = await Denoiser.create({ precision: 'fp16', quality: 'fast' });
const renderer = new THREE.WebGPURenderer({ device: denoiser.device });
```

You can no longer hand the denoiser a canvas/device — onnxruntime-web creates
and owns the `GPUDevice` ([#26107](https://github.com/microsoft/onnxruntime/issues/26107)).
Share `denoiser.device` with your renderer instead (create the denoiser first).

### Executing

```ts
// v1 — stateful: set modes/inputs, execute, receive via listener
denoiser.inputMode = 'imgData';
denoiser.outputMode = 'imgData';
denoiser.onExecute((result) => draw(result), 'imgData');
await denoiser.execute(colorImg, albedoImg, normalImg);

// v2 — one call, result returned
const img = await denoiser.denoise(colorImg, { albedo: albedoImg, normal: normalImg });
ctx.putImageData(img, 0, 0);
```

### Method-by-method

| v1 | v2 |
|---|---|
| `execute(color, albedo, normal)` | `denoise(color, { albedo, normal })` → `ImageData` |
| `setInputImage(name, img, flipY)` | pass inputs directly to `denoise()` (accepts ImageData, img/canvas/video elements, ImageBitmap, or raw `{data,width,height}`) |
| `setInputData(name, f32/u8, w, h)` | same — the `ImgInput` union covers raw RGBA8 |
| `setInputTexture(name, WebGLTexture)` | **gone (WebGL)** → `denoiseTextures({ color, albedo, normal })` with `GPUTexture`s |
| `setInputBuffer(name, GPUBuffer)` | `denoiseTextures(...)` (textures replaced buffers as the GPU interchange) |
| `setInputTensor(name, tf.Tensor)` | gone with TFJS |
| `inputMode` / `outputMode` | gone — the method called *is* the mode (`denoise` → ImageData, `denoiseToFloat` → floats, `denoiseTextures` → GPUTexture) |
| `setCanvas(canvas)` / `outputToCanvas` | gone — `putImageData` the result, or blit the output texture yourself |
| `onExecute(cb, returnType)` | `on('executed', cb)`, or just `await` the call |
| `onProgress(cb)` | `on('progress', cb)` or per-call `onProgress` |
| `onBackendReady(cb)` | unnecessary — `await Denoiser.create()` |
| `abort()` | `abort()` (unchanged: in-flight call resolves `undefined`) |
| `dispose()` | `dispose()` frees buffers/extra sessions but **keeps the device**; full teardown is `destroyDevice()` (⚠ kills the shared device and everything on it) |

### Properties → per-call options

| v1 instance flag | v2 |
|---|---|
| `quality` | `Denoiser.create({ quality })`, still mutable via `denoiser.quality` |
| `hdr` | `denoiseTextures({ hdr: true })` per call |
| `srgb` | `denoise(img, { srgb })` (default `true` for image inputs); texture path uses `transfer: 'linear' \| 'srgb' \| 'aces-srgb'` for output encoding |
| `flipOutputY` | `flipY` (`denoise`) / `outputFlipY` (`denoiseTextures`); input orientation via `inputFlipY` / `auxInputFlipY` |
| `height` / `width` | gone — taken from the input |
| `useTiling` / `tileSize` / `batchSize` | automatic: whole-frame up to `maxRunPixels` (create option), tiled above it with `batch` tiles per run |
| `weightsPath` / `weightsUrl` | `Denoiser.create({ weightsUrl })` — now points at `.onnx` models, not `.tza` |
| `cleanAux` / `dirtyAux` | automatic — rasterized aux is treated as clean when both albedo+normal are present |
| `debugging` / `stats` | `denoiser.stats` (per-stage timings of the last run) |

### Errors

v2 throws typed errors instead of generic ones: `DenoiserUnsupportedError`
(WebGPU/feature availability) and `DenoiserInputError` (mismatched sizes,
missing aux, bad formats).

## Weights hosting

v1 served `.tza` files from a `/tzas` directory or CDN. v2 serves converted
`.onnx` models (`rt_ldr.onnx`, `rt_hdr_alb_nrm_small.fp16.onnx`, …) the same
way — default CDN, override with `weightsUrl`. Regenerate models from OIDN
`.tza` weights any time with [`tools/onnx-convert`](../../tools/onnx-convert/README.md)
(Python, no PyTorch).

## Output differences to expect

- v2's output parity vs native OIDN was tightened during the port (PU transfer,
  autoexposure, normal encoding, no final activation —
  [`docs/specs/oidn-color-reference.md`](../specs/oidn-color-reference.md)), so
  v2 output is *more* faithful to upstream OIDN than v1 was — i.e. not
  bit-identical to v1.
- fp16 (`precision: 'fp16'`) measures ≈ 53 dB PSNR vs fp32 — visually identical,
  ~15% faster.
