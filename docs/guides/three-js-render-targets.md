# three.js: render targets in, render targets out

The zero-copy integration path: get a three.js render target **into** the
denoiser as a `GPUTexture`, and get the denoised result **out** into a texture
three.js can sample — no CPU pixels anywhere. Everything here is lifted from the
working example ([`examples/three-pathtracer-webgpu/src/main.ts`](../../examples/three-pathtracer-webgpu/src/main.ts)).

Requires three ≥ r185 (`three/webgpu`, the `WebGPURenderer`).

## 0. One GPUDevice, created by the denoiser

onnxruntime-web creates and owns the `GPUDevice` and ignores an injected one
([onnxruntime#26107](https://github.com/microsoft/onnxruntime/issues/26107)).
So the order is fixed: **denoiser first, then hand its device to three.js**.

```ts
import * as THREE from 'three/webgpu';
import { Denoiser } from 'denoiser';

const denoiser = await Denoiser.create({ precision: 'fp16' });

const renderer = new THREE.WebGPURenderer({ canvas, device: denoiser.device });
await renderer.init();
```

The denoiser requests the adapter's max limits/features (scoped patch, restored
immediately), so the shared device can hold whole-frame U-Net intermediates and
your renderer's pipelines.

**Lifetime rules** (consequences of ORT owning the device):

- `denoiser.dispose()` — safe between workloads: frees buffers and extra model
  sessions, **keeps the device alive**.
- `denoiser.destroyDevice()` — releasing the last ORT session **destroys the
  shared device**; your renderer, canvas contexts, and every resource on it die
  with it. Only call when the whole WebGPU stack is going away.

## 1. Getting a render target IN

### Unwrap the three.js texture to a raw `GPUTexture`

three.js doesn't expose the backing `GPUTexture` publicly; it lives in the
backend's data map:

```ts
const backendGet = (threeTexture: THREE.Texture): GPUTexture | undefined =>
  (renderer.backend as any).get(threeTexture)?.texture;

const colorTex = backendGet(myRenderTarget.texture);
```

Two caveats, both learned the hard way:

- **This is three internal API** — pin your three version and re-check on
  upgrades.
- **Fetch it fresh each use.** Render targets are re-created on resize (and
  e.g. the path tracer swaps its output target on reset), so a cached
  `GPUTexture` can go stale/destroyed. Resolve from the three texture every
  frame you use it.

### Input requirements

| requirement | detail |
|---|---|
| format | float texture (`rgba16float` / `rgba32float`), **linear** values |
| LDR vs HDR | LDR: values in [0,1], omit `hdr`. HDR: pass `hdr: true` — OIDN's PU transfer + autoexposure are applied around the model (or pin exposure with `inputScale`) |
| orientation | compute-written targets (path tracer `StorageTexture` output) are **bottom-up** → `inputFlipY: true`. Rasterized targets (an MRT G-buffer pass) are top-down → `auxInputFlipY: false`. If a result is upside down, toggle these first |
| size | anything; up to `maxRunPixels` (default 1080p-ish) runs whole-frame in one model pass, larger tiles automatically |

### Optional but strongly recommended: aux G-buffer (albedo + normal)

OIDN's guided models dramatically improve low-sample quality. Rasterize the
same scene once into an MRT target — albedo = unlit base color, normal =
view-space normals in [-1,1]:

```ts
import { mrt, diffuseColor, normalView } from 'three/tsl';

const gbuffer = new THREE.RenderTarget(W, H, { count: 2, type: THREE.HalfFloatType });

renderer.setMRT(mrt({ albedo: diffuseColor, normal: normalView }));
renderer.setRenderTarget(gbuffer);
renderer.render(scene, camera);
renderer.setRenderTarget(null);
renderer.setMRT(null);

const albedo = backendGet(gbuffer.textures[0]);
const normal = backendGet(gbuffer.textures[1]);
```

Rasterized aux is noise-free, so the higher-quality `cleanAux` (`*_calb_cnrm`)
model is selected automatically when both planes are present. The G-buffer is
view-dependent — re-render it when the camera moves.

## 2. Running the denoiser

```ts
const outTex = await denoiser.denoiseTextures({
  color: colorTex,        // GPUTexture, float, linear
  albedo, normal,         // optional GPUTextures -> guided model
  hdr: true,              // linear-HDR input (path tracer output)
  inputFlipY: true,       // tracer target is bottom-up
  auxInputFlipY: false,   // raster G-buffer is already top-down
  transfer: 'linear',     // see output section
  output: myStorageTex,   // optional caller-owned destination
});
if (!outTex) return;      // aborted mid-flight (e.g. camera moved)
```

### `denoiseTextures` parameters (complete)

| param | type | default | meaning |
|---|---|---|---|
| `color` | `GPUTexture` | required | noisy beauty, float linear (HDR or LDR) |
| `albedo` | `GPUTexture` | – | aux plane, [0,1] floats |
| `normal` | `GPUTexture` | – | aux plane, [-1,1] floats (network encoding handled internally) |
| `hdr` | `boolean` | `false` | apply OIDN PU transfer + autoexposure around the model |
| `inputScale` | `number` | auto | manual HDR exposure scale (overrides autoexposure) |
| `inputFlipY` | `boolean` | `false` | color rows are bottom-up (WebGPU render targets) |
| `auxInputFlipY` | `boolean` | = `inputFlipY` | aux orientation when it differs from color |
| `output` | `GPUTexture` | – | caller-owned destination (`rgba8unorm` or `rgba16float`, must have `STORAGE_BINDING`) |
| `transfer` | `'linear' \| 'srgb' \| 'aces-srgb'` | `'linear'` | output encoding — see below |
| `outputFlipY` | `boolean` | `false` | flip the result vertically |
| `onProgress` | `(p: number) => void` | – | tile progress (only fires in tiled mode) |

Returns `GPUTexture` — your `output` if given, else an **engine-owned
`rgba8unorm` texture valid until the next call / size change / teardown** — or
`undefined` if `abort()` was called mid-run.

### `Denoiser.create` options (complete)

| option | type | default | meaning |
|---|---|---|---|
| `precision` | `'fp32' \| 'fp16'` | `'fp32'` | fp16 models + tensors + WGSL IO when `shader-f16` is available (auto-falls back). ~15% faster, PSNR ≈ 53 dB vs fp32 |
| `quality` | `'fast' \| 'balanced' \| 'high'` | `'fast'` | model family: fast = `*_small`, balanced = base, high = `*_large` where available. Mutable later via `denoiser.quality` |
| `weightsUrl` | `string` | jsDelivr CDN | where the `.onnx` models are served |
| `wasmPaths` | `string` | jsDelivr CDN | where ORT's wasm assets load from |
| `maxRunPixels` | `number` | `2048*1152` | above this, tile instead of whole-frame |
| `batch` | `number` | `8` | max tiles per model run in tiled mode |
| `graphCapture` | `boolean` | `false` | opt-in ORT graph capture — **leave off** (no measured gain; ORT 1.27 crashes after ~150–250 replays) |

## 3. Getting the result OUT (three renders it)

### Option A — engine-owned texture (simplest)

Call without `output`; blit or sample the returned `rgba8unorm` texture. With
`transfer: 'aces-srgb'` it's display-ready — e.g. copy straight to a canvas:

```ts
ctx.configure({ device: denoiser.device, format: 'rgba8unorm',
                usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
const enc = denoiser.device.createCommandEncoder();
enc.copyTextureToTexture({ texture: outTex }, { texture: ctx.getCurrentTexture() },
                         { width: outTex.width, height: outTex.height });
denoiser.device.queue.submit([enc.finish()]);
```

Remember it's reused: copy it out (or render it) before the next `denoiseTextures` call.

### Option B — resolve into a texture three.js owns (composable)

Create a `StorageTexture`, register it with the renderer, unwrap it, and pass
it as `output`. Then it's an ordinary three.js texture — feed it to materials,
post-processing (the FSR1 upscaler in the example samples it), anything:

```ts
const denoisedTex = new THREE.StorageTexture(W, H);
denoisedTex.type = THREE.HalfFloatType;          // rgba16float: unclamped linear HDR
denoisedTex.colorSpace = THREE.LinearSRGBColorSpace;
denoisedTex.generateMipmaps = false;
renderer.initTexture(denoisedTex);               // materialize the GPU resource

const denoisedGpuTex = backendGet(denoisedTex)!; // pass as `output:`
```

Pick `transfer` to match who does color management:

- **`'linear'` + `rgba16float` output + let three tonemap** (recommended):
  the denoiser writes unclamped linear HDR; three's `renderer.toneMapping =
  ACESFilmicToneMapping` + output color space do the rest. No color handling
  baked into the denoise.
- **`'aces-srgb'`**: display-ready bytes; use when the consumer bypasses
  three's transforms (raw blits, FSR1's EASU input contract).
- **`'srgb'`**: sRGB-encode without tonemapping (LDR pipelines).

Note: sRGB texture formats can't be `STORAGE_BINDING` — a caller-owned output
is `rgba8unorm` or `rgba16float`, encoding chosen via `transfer`.

## 4. Live loop pacing (progressive denoise while accumulating)

The pattern from the example's live mode:

```ts
let busy = false;
async function maybeDenoise() {
  if (busy) return;
  busy = true;
  try { await runDenoise(); } finally { busy = false; }
}
controls.addEventListener('change', () => denoiser.abort()); // stale view -> drop result
```

`abort()` doesn't cancel GPU work already submitted; it makes the in-flight call
resolve `undefined` so you never present a result for an outdated camera.
`denoiser.stats` has per-stage timings of the last run for HUDs.

## Pitfalls checklist

- Result upside down → `inputFlipY` / `auxInputFlipY` / `outputFlipY` (compute
  targets bottom-up, raster targets top-down).
- Black output from a fresh target → did you `renderer.initTexture()` before
  unwrapping? Is the tracer target actually rendered yet (width matches)?
- Stale/destroyed texture errors after resize/reset → you cached a `GPUTexture`;
  re-resolve via `backendGet` each use.
- Whole app's WebGPU dies → something called `destroyDevice()` (or released the
  last ORT session). Use `dispose()` between workloads.
- Double-image/ghosting in aux mode → G-buffer not re-rendered after camera move.
