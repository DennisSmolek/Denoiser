![title-card-resized](https://github.com/DennisSmolek/Denoiser/assets/1397052/ffe87fd5-00e6-464e-b8a2-ba80402b9d2f)

## AI Denoising that runs in the browser.

#### Based on [Open Image Denoise (OIDN)](https://github.com/RenderKit/oidn) and powered by WebGPU + [onnxruntime-web](https://onnxruntime.ai/)

Denoiser runs OIDN's pre-trained U-Nets fully on the GPU: ONNX models on the
WebGPU execution provider, with all pre/post-processing (normalization, tiling,
overlap blending, color transforms) as WGSL compute on the same `GPUDevice`.
Images up to ~1080p denoise in a single model run — 512² in ~14 ms and 1080p in
~104 ms warm (fp16, M-series laptop) — fast enough to denoise progressively
while a path tracer accumulates.

> **v2 note:** the library was rewritten from TensorFlow.js (abandoned) to
> WebGPU + onnxruntime-web, with a new **stateless per-call API** (clean break).
> Coming from 0.x? See the
> [migration guide](docs/guides/migrating-from-v1.md).

### Basic example

```ts
import { Denoiser } from "denoiser";

const denoiser = await Denoiser.create();
const clean = await denoiser.denoise(document.getElementById("noisy-img")); // ImageData
canvas.getContext("2d").putImageData(clean, 0, 0);
```

### Zero-copy GPU pipeline (three.js)

Create the denoiser first (onnxruntime-web owns the `GPUDevice`), hand
`denoiser.device` to your `WebGPURenderer`, and run render targets straight
through — pathtracer → denoiser → render target, no CPU pixels:

```ts
const denoiser = await Denoiser.create({ precision: "fp16" });
const renderer = new THREE.WebGPURenderer({ device: denoiser.device });

const out = await denoiser.denoiseTextures({
  color: tracerGpuTexture,   // float, linear HDR
  albedo, normal,            // optional MRT G-buffer -> guided model auto-selected
  hdr: true,
  inputFlipY: true,          // render targets are bottom-up
  output: myStorageTexture,  // optional: resolve into a texture three.js samples
});
```

**Docs:**
- [`packages/denoiser/README.md`](packages/denoiser/README.md) — install, API sketch, device-lifetime rules.
- [`docs/guides/three-js-render-targets.md`](docs/guides/three-js-render-targets.md) — render targets in/out, full parameter reference, pitfalls.
- [`docs/guides/migrating-from-v1.md`](docs/guides/migrating-from-v1.md) — 0.x → 2.x mapping.
- [`docs/`](docs/README.md) — index: status/next actions, specs, perf results.

### Examples (`/examples`)

- `three-pathtracer-webgpu` — three r185 `WebGPUPathTracer` → denoiser on one
  shared device: CPU vs zero-copy paths, live progressive denoising, MRT
  G-buffer aux.
- `bench` — the performance harness (sizes × precision × quality, PSNR parity).
- `denoiser-package-test` — minimal end-to-end package test.
- `webgpu-ort-smoke` — low-level ORT/WGSL engine harness.

### Repo layout

- `packages/denoiser` — the library (npm: `denoiser`).
- `tools/onnx-convert` — Python tooling that builds the ONNX U-Nets directly
  from OIDN `.tza` weights (no PyTorch). Models ship fp32 + fp16 with dynamic
  batch/height/width dims. They are hosted separately from the library (a page
  only fetches the one model it uses, 0.6–15 MB) — see
  [Hosting the models](packages/denoiser/README.md#models--weights).
- `docs/` — guides (three.js interop, v1→v2 migration), status/next actions,
  specs, and the perf plan + measured results.

### Development

```sh
corepack yarn install
cd packages/denoiser && yarn build

# convert models once (serves at /models in the example dev servers)
cd tools/onnx-convert && python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python convert.py ../../packages/denoiser/tzas/*.tza -o ../../packages/denoiser/models
python convert.py ../../packages/denoiser/tzas/*.tza -o ../../packages/denoiser/models --fp16

cd examples/three-pathtracer-webgpu && yarn dev
```

### License

MIT. OIDN weights are Apache-2.0 (© Intel) — see `packages/denoiser/tzas/LICENSE.txt`.
