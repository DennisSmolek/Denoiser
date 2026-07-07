# ldraw-eiffel — orbit / path trace / denoise

A remake of the old-pathtracer demo scene on the WebGPU stack: a table surface,
an HDRI, a gelatinous cube, and the LDraw LEGO Architecture **Eiffel Tower
(21019)** — with a [camera-controls](https://github.com/yomotsu/camera-controls)
orbit camera.

The interaction model:

- **camera moving** → plain raster render (the "diffuse" preview)
- **camera at rest** → the WebGPU path tracer accumulates up to *max samples*
- **accumulation done** → the denoiser resolves the frame and it fades in on top
  (check *denoise while accumulating* to re-denoise on every sample instead)

The compare slider under the canvas reveals raw accumulation vs denoised.

```sh
yarn workspace ldraw-eiffel dev   # http://localhost:5177
```

Same architecture as `examples/three-pathtracer-webgpu`: the denoiser (ORT)
creates the GPUDevice, three.js borrows it, and everything — tracing, G-buffer
aux, denoise — stays on that one device with zero CPU readbacks.

## Assets

- `public/assets/eiffel-tower.mpd` — the source model from
  [gkjohnson/ldraw-parts-library](https://github.com/gkjohnson/ldraw-parts-library)
  (mirrors the LDraw OMR; CCAL 2.0).
- `public/assets/eiffel-tower_Packed.mpd` — standalone build of the above with
  all 221 referenced parts + LDConfig colors inlined, so the page never hits a
  parts-library CDN. Regenerate with `yarn workspace ldraw-eiffel pack-model`
  (see `tools/pack-ldraw.mjs`, a CDN-backed port of three.js'
  `packLDrawModel.mjs`). To use a different model, drop its `.mpd` in
  `public/assets/` and point the pack script (and `src/main.ts`) at it — the
  old Tokyo skyline set works the same way.
- `public/assets/kloppenheim_06_puresky_1k.hdr` — Poly Haven HDRI (CC0), the
  default: a soft overcast sky whose near-uniform luminance converges cleanly.
- `public/assets/brown_photostudio_02_1k.hdr` — Poly Haven HDRI (CC0), kept as
  a stress test: its small intense lamps stay salt-and-pepper noisy in this
  tracer branch (no concentrated-source sampling yet) and the speckle survives
  the denoiser. Swap the path in `src/main.ts` to see it.

## Debugging the denoiser inputs

The **view** selector overlays the network's actual inputs: *color* (with the
engine's `inputFlipY` applied), *albedo*, and *normal* — exactly as the network
pairs them, so a wrong flip flag shows up as vertical mismatch between views.
Display transforms only: color is Reinhard+gamma, albedo gamma, normal raw
`[-1,1] → [0,1]` (wrong input ranges read as washed-out gray / hard clipping).

The G-buffer follows OIDN's aux conventions, which needs **two raster passes**:

- **albedo**: background ON (the env color *is* the correct albedo for
  environment pixels; the engine clamps albedo to [0,1] since HDR env values
  can exceed it), transparent surfaces through-blend — or first-hit with the
  *opaque aux* toggle.
- **normal**: background OFF (cleared target ⇒ normal = 0, OIDN's env
  convention — the env quad's `normalView` is a meaningless gradient
  otherwise), transparency always off (alpha-blended normals are garbage;
  first-hit normals are the contract).

Append `?inspector` for the official three.js Inspector (r180+): render-target
viewer, node parameters, profiler — `renderer.inspector = new Inspector()`.

## Notes / upstream quirks

- The tracer stays at 512×512: the unreleased WebGPUPathTracer branch wedges on
  `setSize`/`renderScale` (same constraint as the other example).
- Transmission/refraction isn't implemented in the WebGPU tracer yet, so the
  gel cube uses `transparent` + `opacity` (stochastic alpha) for the traced
  look and `transmission` for the raster preview.
- The LDraw model is merged into a single multi-group mesh
  (`LDrawUtils.mergeObject`) — one BVH build instead of one per brick.
