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

## Quick start (image in, canvas out)

```ts
import { Denoiser } from 'denoiser';

const denoiser = new Denoiser();
denoiser.setCanvas(document.getElementById('output-canvas'));
await denoiser.execute(noisyImageElement); // ImageData / img / canvas / bitmap
```

## Zero-copy GPU pipeline (three.js / render targets)

The integration path for renderers: **pathtracer → denoiser → render target**,
no CPU pixels anywhere.

```ts
const denoiser = new Denoiser();
await denoiser.build();

// ORT creates the GPUDevice — share it with three.js (see Device sharing below)
const renderer = new THREE.WebGPURenderer({ device: denoiser.device });

// input: your renderer's float, linear-HDR output texture
denoiser.hdr = true;
denoiser.setInputTexture('color', tracerGpuTexture);

// output: a texture YOU own (e.g. a three.js StorageTexture's GPUTexture).
// rgba16float = unclamped linear HDR out — your tonemapping stays in charge.
// rgba8unorm = clamped display-ready (optionally tonemapOutput for ACES+sRGB).
denoiser.setOutputTexture(threeStorageGpuTexture);
await denoiser.execute();
// ...render the storage texture like any other three.js texture
```

At 512×512 this runs in ~14–20 ms warm on an M-series laptop — fast enough to
denoise progressively **while a path tracer accumulates**. See
`examples/three-pathtracer-webgpu` (live-denoise checkbox) for the full loop.

### Aux inputs (albedo + normal)

OIDN's aux-guided models sharply improve quality at low sample counts. Rasterize
albedo (base color) and view-normals into float textures (e.g. a three.js MRT
target) and add them — the matching model is selected automatically:

```ts
denoiser.setInputTexture('albedo', albedoGpuTexture); // [0,1] floats
denoiser.setInputTexture('normal', normalGpuTexture); // [-1,1] floats
```

Rasterized aux is noise-free, so the `cleanAux` (`calb_cnrm`) models apply.
CPU-side aux via `setInputData`/`setInputImage` works too (RGBA8; normals get
mapped from [0,1] to [-1,1]).

## How it runs fast

- **Whole-frame inference**: images up to ~1080p (configurable) run through the
  U-Net in ONE `session.run` — no tiling, no overlap redundancy, no seams. The
  ONNX models export with named dynamic dims pinned per session.
- **Adaptive tiling**: bigger images fall back to 1024/512/256 tiles (batched
  per run, sigmoid overlap blending) based on a pixel budget and the device's
  buffer limits.
- **fp16 end-to-end**: `new Denoiser({ precision: 'fp16' })` uses fp16 models,
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

1. **`denoiser.dispose()` can destroy the shared device.** When the last ORT
   session is released, ORT destroys its device — three.js, canvas contexts,
   and every resource on it die with it (`GPUDevice.lost` fires with reason
   `'destroyed'`). Only call `dispose()` on full teardown.
2. Model switches are safe: `Denoiser` overlaps the new engine's creation with
   the old one's disposal precisely so the session count never hits zero
   mid-swap. Don't "optimize" that ordering away.

The denoiser also patches the device request (scoped, restored immediately) so
the device gets the adapter's **max limits + features** — ORT alone requests a
minimal device that can't hold whole-frame U-Net intermediates or a path
tracer's pipelines.

## API sketch

| | |
|---|---|
| `new Denoiser(opts?)` | `precision: 'fp32'\|'fp16'`, `batch`, `wasmPaths`, `graphCapture` |
| `execute(color?, albedo?, normal?)` | run; returns per `outputMode` (or `undefined` with listeners) |
| `setInputImage / setInputData` | DOM images / raw RGBA8 |
| `setInputTexture(name, gpuTexture)` | zero-copy float texture input |
| `setOutputTexture(gpuTexture?)` | resolve into your texture (rgba8unorm / rgba16float, STORAGE_BINDING) |
| `outputMode` | `'imgData'` (default) \| `'float32'` \| `'gpuTexture'` |
| props | `quality: 'fast'\|'balanced'\|'high'`, `hdr`, `srgb`, `flipInputY`, `flipOutputY`, `tonemapOutput` |
| events | `onExecute`, `onProgress`, `onBackendReady` |
| info | `device`, `lastStats` (per-stage timings), `tileInfo` |

Orientation: WebGPU render targets read bottom-up. Either set `flipInputY`
(normalize on read) or leave the pipeline bottom-up and let your renderer's UVs
display it upright — don't set both.

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
