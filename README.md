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
> WebGPU + onnxruntime-web. The high-level API is largely compatible; the
> WebGL/TFJS backends are gone.

### Basic example

```ts
import { Denoiser } from "denoiser";

const denoiser = new Denoiser();
denoiser.setCanvas(document.getElementById("output-canvas"));

async function doDenoise() {
  await denoiser.execute(document.getElementById("noisey-img"));
}
```

### Zero-copy GPU pipeline (three.js)

Share one `GPUDevice` between your renderer and the denoiser, feed a float
render target in via `setInputTexture`, and resolve straight into a texture you
own via `setOutputTexture` — pathtracer → denoiser → render target, no CPU
pixels. Aux inputs (albedo + view-normals from an MRT pass) select OIDN's
higher-quality guided models automatically.

**Full API documentation: [`packages/denoiser/README.md`](packages/denoiser/README.md)** —
including the important notes on device sharing/lifetime with onnxruntime-web.

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
  batch/height/width dims.
- `perf-plan.md` — the performance work: plan, phases, measured results.

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
