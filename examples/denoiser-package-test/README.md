# denoiser package — end-to-end test

Validates the rewritten WebGPU/ONNX `denoiser` package through its public API
(`import { Denoiser } from 'denoiser'`), not the engine internals. Exercises model
selection (`rt_ldr_small`), image input, tiling (384² → padded 256² tiles), and
canvas output.

## Prereqs

1. Models converted into `packages/denoiser/models/` (see `tools/onnx-convert/`).
2. The package built: `yarn workspace denoiser build` (the example imports its `dist`).
3. A WebGPU browser.

## Run

```sh
yarn dev      # MUST be dev — vite.config serves /models/* only in dev (configureServer)
```

Open the URL: left = noisy 384² image, right = denoised (written to canvas by
`denoiser.setCanvas`). The status box logs timing and confirms `denoiser.device`
is exposed (the GPUDevice to share with three.js in Phase 2).

`weightsUrl` is pointed at the dev `/models` route; in production set it to your CDN.
