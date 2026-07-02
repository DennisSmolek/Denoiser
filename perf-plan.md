# Performance Plan — WebGPU/ONNX Denoiser

Review of the current pipeline (`packages/denoiser/src/ort/engine.ts` + `ort/wgsl.ts`),
where the time actually goes, and a phased plan. Facts about onnxruntime-web below were
verified against the v1.27.0 source and current docs (July 2026).

## Where the time goes today

For an image of W×H with tile 256 / overlap 32 (stride 224):

| Cost | Notes |
|---|---|
| Per-tile serialization | Each tile is `submit(extract)` → `await session.run()` → `submit(accumulate)`. Every tile is a full JS↔GPU sync point; the GPU idles between tiles. 1080p = 45 tiles. |
| Per-run CPU dispatch | Without graph capture, ORT re-walks the graph and re-encodes every kernel dispatch (dozens for the U-Net) in WASM/JS on every `run()`. |
| Overlap redundancy | 256²/224² ≈ **1.31× pixels inferred** — a fixed ~30% tax, plus tile-count inflation. |
| CPU boundary | Input is `Uint8ClampedArray` RGBA8 written from JS; in the pathtracer example the frame does GPU→CPU float readback → JS tonemap loop → RGBA8 → GPU. Dwarfs inference at real sizes, and quantizes HDR to 8-bit before the model sees it. |
| fp32 everywhere | Tensors, WGSL buffers, and (in practice) models are all fp32 — 2× the bandwidth of fp16 on a conv net that is largely bandwidth-bound. |

Honest ceiling: without WebNN the browser can't touch ANE/AMX/XMX/tensor-core paths that
native OIDN uses. But the gap between ~5–7ms/256-tile × N tiles and native is mostly
*orchestration* overhead, which we can remove.

## Bugs / review notes found along the way

1. **fp16 models are currently unusable** (`Models.precision = 'fp16'` fetches `.fp16.onnx`,
   whose graph IO is FLOAT16, but `engine.ts` always creates `dataType: 'float32'`
   gpu-buffer tensors → type mismatch at run time). Fix in Phase 3 (or guard sooner).
2. **Engine buffer reuse is already graph-capture-shaped** (lucky!): ORT's graph capture
   binds IO **once on the first run and never re-binds** — passing a *different* GPUBuffer
   later is **silently ignored**. Our engine reuses the same `nchwInput`/`outNCHW` and
   mutates contents in place, which is exactly the supported pattern. Keep it that way;
   never swap those buffer objects per-run.
3. Bind groups are recreated per dispatch and each op shares ONE uniform buffer
   (`extractParams` etc.), which forces write→submit per tile and blocks batching
   multiple tiles into one encoder. Needs per-tile param slots (dynamic offsets or a
   param array indexed by dispatch).
4. `handleReturn`'s `float32` output mode does a per-pixel JS loop; `flipOutputY` is a CPU
   pass — both foldable into the `resolve` kernel (flip flag / float output). Minor.
5. When batching accumulates into one encoder: overlapping tiles read-modify-write the
   same `accum`/`weight` regions. Keep each accumulate in its **own compute pass**
   (pass boundaries synchronize storage access); don't merge dispatches into one pass.
6. `onnxruntime-web/webgpu` import = the **native (C++) WebGPU EP** bundle in 1.27 — the
   right one (JSEP is being phased out; new perf work lands in the native EP). Keep.

## The plan

Ordering rationale: measure first; then the one-line win (capture); then kill the
per-tile sync (batching / fewer-bigger tiles — these amortize the same overhead, do the
cheap one first); then fp16 (multiplies with everything above); then zero-copy IO
(biggest real-world win for the pathtracer, and it's already TODO 2b).

### Phase 0 — Benchmark harness + profiling (foundation, ~half day)
- Add a bench page (extend `denoiser-package-test`): fixed seeds/images at 512², 1080p;
  report cold/warm, per-stage timers (upload / per-tile extract / run / accumulate /
  resolve / readback) via `performance.now()` + GPU timestamps where available.
- Turn on ORT profiling/verbose once (`ort.env.logLevel = 'verbose'`,
  `enableProfiling`) and **confirm every op runs on the WebGPU EP** — one CPU-fallback
  op silently destroys everything downstream. (Graph capture in Phase 1 doubles as an
  assertion: session creation fails if any op falls off the EP.)
- Record the TFJS-era number and native OIDN (if available) for the same image as
  reference lines. Every later phase must post numbers against this.

### Phase 1 — Graph capture (~1 line + fallback, immediate)
- `enableGraphCapture: true` in `DenoiseEngine.create`, wrapped in try/catch → retry
  without it (creation throws if the model/EP combo doesn't qualify).
- Requirements already met: static shapes ✓, gpu-buffer input tensor ✓,
  `preferredOutputLocation: 'gpu-buffer'` ✓, stable reused buffers ✓ (see note 2).
- Watch: we run **one session per model but share the device with three.js** — capture
  replays command buffers on the shared queue; interleaved three.js submits should be
  queue-ordered and fine, but verify visually. 1.27.0 specifically stabilized
  multi-graph-capture buffer management, so land on ≥1.27.

### Phase 2 — Kill the per-tile sync: batching + fewer submits (~2–3 days, biggest architectural win)
- **Converter**: emit *named* dynamic dims — `["batch", C, "height", "width"]`
  (`freeDimensionOverrides` needs named free dims). One artifact per weight set instead
  of per-size; pin `{batch, height, width}` per session at creation. Keep
  `--size` as a fallback for fully-static export.
- **Engine**: create the session with `freeDimensionOverrides: {batch: B, height: 256, width: 256}`
  (B ≈ 4–8; benchmark) + graph capture. One big NCHW input buffer `[B, C, 256, 256]`,
  one output `[B, 3, 256, 256]`.
- **WGSL**: extend `extractTile` to write tile *b* at batch offset (dispatch Z = batch
  index, per-tile params in a storage array); same for accumulate (separate passes per
  tile, note 5). Encode ALL extracts in one encoder/submit, `run()` once per batch,
  all accumulates in one submit. Remainder tiles: pad the batch with duplicate tiles
  and skip their accumulate (capture freezes B, so no per-run shape changes).
- Also: cache bind groups (buffers are stable now), merge the clear submits into the
  first encoder.
- Expected: per-image `await` count drops from `tiles` to `ceil(tiles/B)`; GPU stays fed.

### Phase 3 — True fp16 end-to-end (~1–2 days)
- Fix note 1: pick tensor `dataType` from `Models.precision`; halve `nchwInput`/`outNCHW`
  sizes.
- WGSL: `enable f16;` in extract/accumulate when the device has `shader-f16` (ORT already
  requests it on its device when available — check `device.features`), write/read `f16`
  tile buffers; keep `accum`/`weight`/resolve in f32.
- Validate parity vs fp32 on the bench page (fp16 conv overflow bugs exist in ORT for
  other archs; U-Net activations are small, low risk, but measure PSNR).
- Fallback: if `shader-f16` absent, stay on fp32 models.
- Expected: up to ~2× on bandwidth-bound convs + half the tile-IO traffic. Multiplies
  with Phases 1–2.

### Phase 4 — Tile-size strategy / whole-image runs (~1 day, mostly benchmarking)
- With dynamic dims (Phase 2) sessions can be built at ANY /16-aligned size (4 maxpools
  ⇒ H,W ≡ 0 mod 16). For typical render targets, denoise the **whole frame in one run**
  (pad to /16): zero overlap tax, one `run()` per frame. 1080p fp16 U-Net activations
  are plausibly fine on desktop GPUs; ORT reuses intermediate buffers.
- Policy: try `tile = min(paddedImageSize, budget)` where `budget` starts at e.g. 1024²
  and falls back to 512/256 on device limits or allocation failure; keep 256+overlap as
  the floor. Session cache keyed by (model, size, batch).
- Expected at 1080p: 45 tiles → 1–4 runs and ~30% less inferred area.

### Phase 5 — Zero-copy GPU IO (TODO 2b, ~2–4 days, biggest end-to-end win for the pathtracer)
- Input: `denoiser.setInputBuffer/Texture(name, GPUBuffer|GPUTexture, {format})` —
  extract kernel variant that samples a float/rgba16f texture (or reads a float buffer)
  directly: no readback, no JS tonemap, no 8-bit quantization, real HDR into the `hdr`
  models (use OIDN's HDR transfer/scale semantics rather than tonemapped-LDR input).
- Output: optional `GPUBuffer`/`GPUTexture` output mode (resolve writes rgba8/16f
  texture; fold `flipOutputY` in here) → three.js displays it without `putImageData`.
  Readback stays available for image-export workflows.
- This also unblocks TODO 2c (TSL MRT G-buffer → 9ch aux) with albedo/normal as float
  textures instead of 8-bit arrays.

### Phase 6 (parallel/afterwards) — WebNN track
See below — independent of Phases 1–5 (same ONNX artifacts).

## WebNN track — where to start

Status (checked July 2026): WebNN in Chrome/Edge is **still behind a flag**
(`about://flags` → `#web-machine-learning-neural-network`; Windows backend = ONNX
Runtime/Windows ML with NPU support, macOS = CoreML incl. ANE, Linux = CPU-only TFLite).
Spec hit W3C Candidate Rec Jan 2026. So: ship it as an experimental backend, not the default.

The good news — this is much less work than it sounds:
1. **Same models.** ORT-web's WebNN EP consumes our existing ONNX files. Op coverage
   verified against `webnn-operators.md`: Conv→conv2d, Clip→clamp, MaxPool→maxPool2d,
   Concat→concat, Resize→resample2d (nearest, constant scales — our converter uses a
   constant `up_scales` initializer ✓, 4D ✓). The whole U-Net should map with zero
   WASM fallback.
2. **Different bundle + EP config.** WebNN needs `onnxruntime-web/all` (the webgpu
   bundle excludes it): `executionProviders: [{ name: 'webnn', deviceType: 'gpu'|'npu',
   powerPreference: 'default' }]`.
3. **No WebGPU interop yet.** MLTensor↔WebGPU sharing is spec-work-in-progress and NOT
   in ORT 1.27 — so the WGSL pre/post pipeline can't feed WebNN. First cut uses CPU
   tensors (typed-array NCHW pre/post in JS — port of the WGSL math), and optionally
   `Tensor.fromMLTensor` + `preferredOutputLocation: 'ml-tensor'` with a shared
   `MLContext` later. This means WebNN v1 targets the *image/CPU* workflow, not the
   shared-device pathtracer loop. Revisit interop when the MLTensor explainer lands.

### Packaging plan
- **One package, second entry point**: `denoiser/webnn` subpath export (mirrors how we
  already split on ORT bundles). Restructure `src/` into a backend-agnostic core
  (tiling policy, model selection, weights, public `Denoiser` API) + `backends/webgpu`
  (current engine) + `backends/webnn` (new). The webgpu entry keeps importing
  `onnxruntime-web/webgpu`; the webnn entry imports `onnxruntime-web/all`. No bundle-size
  regression for existing users, no second npm package to version.
- `new Denoiser({ backend: 'webnn', deviceType: 'npu' })` on that entry; expose a static
  `Denoiser.webnnAvailable()` (`'ml' in navigator` + context probe) and fall back to
  webgpu with a warning.
- New `examples/webnn-basic`: load an image, toggle deviceType gpu/npu, timing readout
  side-by-side with the webgpu backend. README documents the Chrome flag.

### WebNN steps
1. Core/backends refactor (no behavior change, webgpu path stays verified).
2. CPU pre/post (NCHW extract/blend in TS — small; reuse the tiler math).
3. WebNN engine: session per model with `freeDimensionOverrides` (docs recommend it for
   WebNN), CPU tensors first.
4. Example + flag docs + timings vs webgpu on the bench page (esp. macOS ANE via
   deviceType 'npu' and Windows NPU).
5. Later: MLTensor IO binding; much later: WebGPU interop when it ships.

## Expected outcome

Multiplying the phases at 1080p: no per-tile stalls + captured graph + fp16 + whole-frame
runs should plausibly land **3–8× faster than today's loop**, with the pathtracer path
additionally dropping its ~O(frame) CPU readback/tonemap cost entirely (Phase 5). Native
OIDN will still win on NPU/tensor-core hardware — that gap is what the WebNN track is for.
