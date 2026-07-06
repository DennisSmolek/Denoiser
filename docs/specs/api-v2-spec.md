# API v2 proposal (for discussion)

Problem: the GPU path is configured through instance flags set in the right
order (`hdr`, `srgb`, `flipInputY`, `flipOutputY`, `tonemapOutput`,
`outputMode`, `setInputTexture`, `setOutputTexture`) — stateful, order-
sensitive, and two call sites fighting over the same instance (see the live
demo juggling flags per mode) is a footgun. v2 makes execution stateless-per-
call; the instance holds only identity (models, device, sessions).

## Shape

```ts
const denoiser = await Denoiser.create({
  precision: 'fp16',          // 'fp32' default; auto-falls back
  quality: 'fast',            // 'fast' | 'balanced' | 'high'
  weightsUrl: '/models',
  maxRunPixels: 2048 * 1152,  // whole-frame budget
});

denoiser.device;              // shared GPUDevice (created eagerly by create())

// ---- image workflow (unchanged in spirit) ----
const img = await denoiser.denoise(imageLike, { hdr?, srgb?, flipY? });
// returns ImageData; .denoiseToFloat() for Float32Array

// ---- texture workflow (one call, everything explicit) ----
const out = await denoiser.denoiseTextures({
  color: gpuTexture,          // required; float, linear
  albedo?, normal?,           // aux planes -> model auto-selected
  hdr: true,                  // model family
  inputFlipY?: boolean,
  output?: GPUTexture,        // caller-owned rgba8unorm | rgba16float; else engine-owned
  transfer?: 'linear' | 'srgb' | 'aces-srgb',  // output encoding (replaces tonemapOutput/srgb/flipY pile)
  outputFlipY?: boolean,
  onProgress?: (p) => void,
});                            // returns the GPUTexture written
```

## Lifecycle

- `Denoiser.create()` replaces `new Denoiser()` + implicit `build()` — async
  construction makes device availability explicit, kills the isDirty dance.
- `denoiser.dispose()` — releases sessions/buffers **but keeps the device
  alive** by holding one internal keep-alive session, IF other consumers were
  handed the device (tracked by a `retainDevice()` call or a create flag).
- `denoiser.destroyDevice()` — explicit full teardown: releases everything
  including the device. The dangerous op gets the scary name.
- Model/quality/precision changes: still transparent (engine swap overlaps
  create/dispose internally — the device never drops).

## Misc

- `denoiser.stats` — last-run `DenoiseStats` (unchanged).
- Events object instead of three add/remove pairs: `denoiser.on('progress' | 'executed' | 'ready', cb)`.
- `denoiser-react` rebuilds on top of this core once it settles (hooks:
  `useDenoiser(opts)`, `useDenoisedTexture(colorTex, opts)`).
- Errors: typed (`DenoiserUnsupportedError`, `DenoiserInputError`) so apps can
  branch on WebGPU-missing vs bad-input.

## Back-compat

Ship v2 as the major bump the rewrite already warrants. Keep `execute()` +
flag props working one minor release with deprecation warnings, then drop.
