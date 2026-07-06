# denoiser

**AI denoising in the browser — [Open Image Denoise (OIDN)](https://github.com/RenderKit/oidn) running fully on WebGPU via [onnxruntime-web](https://onnxruntime.ai/).**

Denoiser runs OIDN's pre-trained U-Nets (converted to ONNX) on the WebGPU execution
provider. All pre/post-processing — normalization, layout, tiling, overlap blending,
color transforms — is WGSL compute on the same `GPUDevice`, so the only CPU↔GPU
round-trip is the one you ask for. Feed it images or `GPUTexture`s; get back
`ImageData`, floats, or a texture.

```sh
npm install denoiser onnxruntime-web
```

## Quick start (image in, ImageData out)

```ts
import { Denoiser } from 'denoiser';

const denoiser = await Denoiser.create();
const clean = await denoiser.denoise(noisyImageElement); // ImageData
canvas.getContext('2d').putImageData(clean, 0, 0);
```

## Zero-copy GPU pipeline (three.js / render targets)

The integration path for renderers: **pathtracer → denoiser → render target**,
no CPU pixels anywhere. Execution is stateless — everything about a run is in
the call:

```ts
const denoiser = await Denoiser.create({ precision: 'fp16' });

// ORT creates the GPUDevice — share it with three.js (see Device sharing below)
const renderer = new THREE.WebGPURenderer({ device: denoiser.device });

const result = await denoiser.denoiseTextures({
  color: tracerGpuTexture,     // float, linear HDR
  albedo, normal,              // optional aux planes -> guided model auto-selected
  hdr: true,                   // OIDN PU transfer + autoexposure applied
  inputFlipY: true,            // render targets are bottom-up
  output: threeStorageGpuTexture, // optional caller-owned target
  transfer: 'linear',          // or 'srgb' | 'aces-srgb' (display-ready)
});
// ...render the texture like any other three.js texture
```

At 512×512 this runs in ~14–20 ms warm on an M-series laptop — fast enough to
denoise progressively **while a path tracer accumulates**. See
`examples/three-pathtracer-webgpu` (live-denoise checkbox) for the full loop.

### Aux inputs (albedo + normal)

OIDN's aux-guided models sharply improve quality at low sample counts. Rasterize
albedo (base color) and view-normals into float textures (e.g. a three.js MRT
target) and pass them with the call — the matching model is selected automatically:

```ts
await denoiser.denoiseTextures({
  color,
  albedo: albedoGpuTexture, // [0,1] floats
  normal: normalGpuTexture, // [-1,1] floats
  hdr: true,
});
```

Rasterized aux is noise-free, so the `cleanAux` (`calb_cnrm`) models apply. The
image path takes aux the same way: `denoiser.denoise(color, { albedo, normal })`
(RGBA8; normals encoded [0,1] are mapped to [-1,1] internally).

Full three.js walkthrough (unwrapping render targets, output-into-three,
orientation, pitfalls): [`docs/guides/three-js-render-targets.md`](../../docs/guides/three-js-render-targets.md).
Coming from the 0.x TFJS API: [`docs/guides/migrating-from-v1.md`](../../docs/guides/migrating-from-v1.md).

## How it runs fast

- **Whole-frame inference**: images up to ~1080p (configurable) run through the
  U-Net in ONE `session.run` — no tiling, no overlap redundancy, no seams. The
  ONNX models export with named dynamic dims pinned per session.
- **Adaptive tiling**: bigger images fall back to 1024/512/256 tiles (batched
  per run, sigmoid overlap blending) based on a pixel budget and the device's
  buffer limits.
- **fp16 end-to-end**: `Denoiser.create({ precision: 'fp16' })` uses fp16 models,
  tensors, and WGSL IO (needs the `shader-f16` feature; falls back to fp32).
  ~15% faster, PSNR vs fp32 ≈ 53dB (visually identical).
- Everything between input and output stays on the GPU.

Measured warm (M-series Mac, Chrome, `fast` quality): 512² **13.7 ms**,
720p **45 ms**, 1080p **104 ms** (fp16). `balanced` quality ≈ 2.1× those.

## Device sharing & lifetime — READ THIS if you pass `denoiser.device` around

onnxruntime-web **creates and owns the `GPUDevice`** (it ignores an injected
one — [onnxruntime#26107](https://github.com/microsoft/onnxruntime/issues/26107)).
To share a device with three.js, build the denoiser FIRST and hand
`denoiser.device` to `WebGPURenderer`.

Two lifetime rules follow from ORT's ownership:

1. **`destroyDevice()` destroys the shared device.** When the last ORT session
   is released, ORT destroys its device — three.js, canvas contexts, and every
   resource on it die with it (`GPUDevice.lost`, reason `'destroyed'`).
   `dispose()` is the safe between-workloads cleanup: it frees buffers and
   extra sessions but retains one so the device survives.
2. Model switches are safe: `Denoiser` overlaps the new engine's creation with
   the old one's disposal precisely so the session count never hits zero
   mid-swap. Don't "optimize" that ordering away.

The denoiser also patches the device request (scoped, restored immediately) so
the device gets the adapter's **max limits + features** — ORT alone requests a
minimal device that can't hold whole-frame U-Net intermediates or a path
tracer's pipelines.

## API sketch (v2 — stateless per call)

| | |
|---|---|
| `await Denoiser.create(opts?)` | `precision`, `quality`, `weightsUrl`, `wasmPaths`, `maxRunPixels`, `batch`, `graphCapture` |
| `denoise(imageLike, opts?)` | → `ImageData`; opts: `albedo`, `normal`, `srgb`, `flipY`, `onProgress` |
| `denoiseToFloat(imageLike, opts?)` | → normalized `Float32Array` |
| `denoiseTextures(opts)` | → `GPUTexture`; opts: `color`, `albedo`, `normal`, `hdr`, `inputScale`, `inputFlipY`, `auxInputFlipY`, `output`, `transfer`, `outputFlipY` |
| `abort()` | drop the in-flight run's result |
| `dispose()` | free buffers/extra sessions, KEEP the device |
| `destroyDevice()` | full teardown (kills the shared device) |
| `on('progress' \| 'executed', cb)` | events; returns an off() fn |
| info | `device`, `stats` (per-stage timings), `quality` (mutable) |

Orientation: WebGPU render targets read bottom-up — set `inputFlipY`. If your
aux planes come from a raster pass their convention can differ from a
compute-written color texture; `auxInputFlipY` handles that independently.

`graphCapture` is opt-in and off by default: it measured no gain (the workload
is GPU-bound) and onnxruntime-web 1.27 captured sessions crash after roughly
150–250 cumulative replays regardless of GPU syncs — unusable for live loops
(standalone reproduction: the `ort-webgpu-graphcapture-repro` repo).

## Models / weights

The converted `.onnx` models (fp32 + fp16, all OIDN RT + lightmap variants) load
from a CDN by default; override with `denoiser.weightsUrl = '/models'`. To
regenerate from OIDN `.tza` weights: `tools/onnx-convert` (no PyTorch needed).

## License

MIT. OIDN weights are Apache-2.0 (© Intel) — see `tzas/LICENSE.txt`.
