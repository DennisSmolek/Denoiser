# Upscale pipeline: FSR1 (done) → FSR3-class temporal (roadmap)

> **2026-07-10 — SHIPPED as [`@pmndrs/upscaler`](https://github.com/pmndrs/upscaler)
> (npm `@pmndrs/upscaler@0.1.0`).** The FSR3-class temporal upscaler this doc
> scopes below is now a real, published library: AMD FSR1 (spatial) + FSR2/3-style
> temporal, as raw WGSL compute passes on three's `WebGPURenderer` device, with a
> composable TSL node (`upscale` / `upscaleScene` / `upscaleSpatial`) and an
> imperative `Upscaler` class. It shares the denoiser's `GPUDevice` (grabs
> `renderer.backend.device`) — no extra device, zero-copy.
>
> The new **`examples/upscale-pipeline`** demo wires it into the full
> three.js → denoiser → upscaler chain on one device and **supersedes the FSR1
> stage** baked into `three-pathtracer-webgpu` (`?fsr=1`, three's built-in
> `FSR1Node`). That demo uses the library's **spatial** path — the honest fit for a
> discrete, stills-oriented denoise pass; the temporal path (depth + per-frame
> motion vectors + jitter) is the motion-first integration this doc still describes.
>
> **This document is now historical** — a record of the spike/scoping that led to
> the library. For the shipped API see the `@pmndrs/upscaler` README and the
> `upscale-pipeline` demo.


## Done: FSR1 stage (`examples/three-pathtracer-webgpu?fsr=1`)

Pipeline: pathtracer (512, linear HDR) → denoiser (`hdr` model, `tonemapOutput`
= ACES+sRGB, per EASU's input contract: tonemapped, gamma-encoded, [0,1],
anti-aliased — denoised output satisfies "anti-aliased" naturally) → **three
r184+'s official `fsr1()` TSL node** (EASU+RCAS, MIT, from
`three/addons/tsl/display/FSR1Node.js`) → 1024 canvas. Order matters:
**denoise before upscale** — noise destroys EASU's edge analysis.

Notes / constraints hit:
- `FSR1Node` auto-sizes output to the renderer's drawing buffer; we pin it via
  a `setSize` override and render its node through a `QuadMesh` into an
  rgba8unorm target, then copy to a WebGPU canvas.
- The unreleased WebGPUPathTracer branch **cannot change resolution after
  init** (`setSize`/`renderScale` → permanent reset loop; `renderSample` at
  non-initial sizes hangs the GPU hard enough to starve other tabs). So today
  the demo upscales 512→1024 instead of rendering at 256. When the tracer
  branch fixes resizing (or on scenes we control), the same pipeline renders at
  25-50% display res directly — that's the real speed win
  (denoise cost scales with render pixels: 1080p output at 540p render ≈ 30ms
  total instead of 104ms).

## Next: FSR3-class temporal upscaling (skip FSR2 — agreed)

FSR2 is subsumed by FSR3's upscaler (better reactive-mask handling, improved
disocclusion + thin-feature logic); frame generation (the other half of FSR3)
is out of scope for pathtracing. Target: **FSR3's temporal upscaler only**.

Inputs it requires per frame:
1. **Motion vectors** (per-pixel, screen-space) — from a raster pre-pass
   (three r180+ has velocity infrastructure via `velocity` TSL node / TRAA's
   pipeline) — same pattern as our MRT G-buffer pass, one more target.
2. **Depth** — same raster pass.
3. **Jitter** — subpixel camera offset per frame + jitter sequence fed to the
   upscaler. The path tracer already jitters rays internally; we need it
   *known/controlled* (Halton sequence applied to the camera projection).
4. Color (our denoised output) + optional reactive mask.

Open questions to resolve in a spike:
- **Port scope:** FSR3's upscaler is a large HLSL codebase (~10+ passes:
  reconstruct-prev-depth, depth-clip, locks, accumulate, RCAS). A faithful WGSL
  port is a month-class project. Alternative first step: **three's TRAA +
  bicubic/EASU upsample** as the temporal baseline (already in three, cheap),
  then evaluate whether full FSR3 quality justifies the port.
- **Interaction with accumulation:** a path tracer converging on a still frame
  *is* temporal accumulation; FSR3-class upscaling pays off for *moving*
  cameras/scenes (real-time preview mode), not final stills. The pipeline
  should treat "still refinement" (current demo) and "interactive preview"
  (FSR3 territory) as distinct modes.
- Denoiser placement: temporal upscalers want stable (denoised) color +
  reactive masks where denoising is uncertain — our per-pixel blend weights
  could double as a reactive-mask source (interesting, novel).

Suggested order: (1) motion/depth raster pass (shared with the aux G-buffer),
(2) TRAA-based temporal baseline, (3) FSR3 upscaler port spike (one pass at a
time, verify against the HLSL reference), (4) reactive mask from denoiser
confidence.
