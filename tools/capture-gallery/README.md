# capture-gallery

Dumps the **gallery demo's** input assets (Phase B, B1.3) straight out of the
existing WebGPU pathtracer examples: an spp ladder of noisy color frames plus the
albedo/normal AOVs and a converged reference, exactly as the denoiser network
expects them.

Output goes to `examples/gallery/public/scenes/<sceneId>/`:

| file | what |
|---|---|
| `spp1.png … spp16.png` | noisy color at 1/2/4/8/16 samples, **ACES-tonemapped + sRGB** (the demo's display transform) |
| `reference.png` | converged frame (highest spp reached, ~256) |
| `albedo.png` | albedo AOV, linear **[0,1]** written as raw bytes (no sRGB) — feed it back as-is |
| `normal.png` | view-space normal, **[-1,1] → [0,1]** (`n*0.5+0.5`); env/first-hit=0 → flat 0.5 gray |

and merges `examples/gallery/public/scenes/manifest.json` (schema below).

## Scenes

| id | example | depicts |
|---|---|---|
| `spheres` | `examples/three-pathtracer-webgpu` | 3 RGB spheres on a plane, gradient env |
| `eiffel` | `examples/ldraw-eiffel` | LEGO Eiffel Tower + gel cube on a table, HDRI |

## Run (headless — the normal path)

From the repo root (Node ≥ 21 for native `WebSocket`; macOS Chrome expected at
`/Applications/Google Chrome.app`):

```bash
node tools/capture-gallery/capture.mjs              # all scenes
node tools/capture-gallery/capture.mjs spheres      # one scene
node tools/capture-gallery/capture.mjs eiffel --ref 128
```

Flags: `--ref N` (reference spp target, default 256), `--attempts N` (retries per
scene, default 5), `--headed` (currently still `--headless=new`, reserved).

It spawns each example's vite dev server (which serves the ONNX weights from the
workspace), launches headless Chrome with WebGPU (`--use-angle=metal
--enable-unsafe-webgpu`), drives the page over CDP (no puppeteer), calls
`window.__captureGallery()`, writes the PNGs, and prints independent per-file
pixel stats (dimensions, mean luma, non-black %, 3×3 local variance — the noise
metric that should fall monotonically down the ladder).

The page is loaded with `?headless=1` so the examples skip the compare-overlay
`getContext('webgpu')` that stalls headless Chrome.

### Known risk: nondeterministic init hang

The upstream WebGPU pathtracer branch sometimes wedges during headless init. The
tool retries each scene up to `--attempts` times, killing Chrome + the dev server
and relaunching between tries, with a hard per-attempt timeout. If a scene still
never initializes, use the headed fallback.

## Headed fallback (if headless won't init)

Every example also mounts a **capture button** when you add `?capture=1`:

```bash
yarn workspace three-pathtracer-webgpu dev   # → http://localhost:5173/?capture=1
yarn workspace ldraw-eiffel dev              # → http://localhost:5177/?capture=1
```

Open the URL in a WebGPU browser, let the scene load, click **“capture gallery
assets.”** It runs the same `window.__captureGallery()` and downloads
`spp1.png … spp16.png`, `reference.png`, `albedo.png`, `normal.png`, and
`<id>.manifest.json`. Save the PNGs into
`examples/gallery/public/scenes/<id>/`, then fold the manifest fragment into
`examples/gallery/public/scenes/manifest.json` (an entry in the `scenes` array).

## manifest.json schema

```json
{
  "scenes": [
    {
      "id": "spheres",
      "title": "Spheres (procedural)",
      "width": 512,
      "height": 512,
      "spp": [1, 2, 4, 8, 16],
      "color": { "1": "spp1.png", "2": "spp2.png", "4": "spp4.png", "8": "spp8.png", "16": "spp16.png" },
      "albedo": "albedo.png",
      "normal": "normal.png",
      "reference": "reference.png"
    }
  ]
}
```

All paths are relative to the scene's own folder. Capture size is 512×512.

## How it works / where the logic lives

- `examples/_shared/gallery-capture.ts` — the in-page capture hook
  (`installGalleryCapture`), imported by both examples. It accumulates the spp
  ladder, reads back the tracer's linear-HDR float target and ACES+sRGB-tonemaps
  it in JS (matching the demo's display), and renders **correct** AOV passes:
  albedo with the background on (env color = env albedo, clamped to [0,1]),
  normal with the background **off** and materials forced opaque so env pixels
  read as `normal=0` (OIDN convention) instead of the env quad's gradient — the
  aux bug this tool deliberately avoids.
- `tools/capture-gallery/capture.mjs` — the Node CDP driver (dev server + Chrome
  lifecycle, retries, file writing, manifest merge).
- `tools/capture-gallery/png.mjs` — a tiny dependency-free PNG decoder used only
  to verify the written files.
