# Roadmap & positioning

_Last updated: 2026-07-09. Companion to [STATUS.md](STATUS.md) (tactical next
actions); this doc is the strategic view — where we stand vs v1 and native OIDN,
what the library is actually for, and the phased plan to a public v2 launch._

## Where we stand

### vs v1 (0.0.x, TensorFlow.js)

| | v1 (TFJS) | v2 (ORT-web WebGPU) |
|---|---|---|
| Engine | TFJS WebGL/WebGPU, runtime graph build from TZA | ONNX on WebGPU EP, offline-converted models |
| 512² warm | 37.6 ms (9 tiles) | **13.7 ms** (whole-frame, fp16) — 2.7× |
| 1080p warm | 188 ms (45 tiles) | **104 ms** (whole-frame) — 1.8× |
| Precision | fp32 only usable | fp16 end-to-end (53.5 dB vs fp32) |
| GPU interop | WebGLTexture marshaling, copies | zero-copy texture in/out on a shared `GPUDevice` |
| Aux (albedo+normal) | present, quality never verified | **verified to 1–2 LSB vs native OIDN** (splitAux workaround for the upstream ORT bug) |
| API | stateful (set inputs, execute) | stateless per-call `create`/`denoise`/`denoiseTextures` |
| Browser coverage | WebGL fallback = near-universal | **WebGPU only** — v1 remains the answer for old browsers |
| Maintenance | TFJS is abandoned | onnxruntime-web actively developed; WebNN EP future |

The one true regression is coverage: no WebGL fallback by design. WebGPU is
Chrome/Edge stable, Safari 26+, Firefox (win stable/others in progress) — fine
for a graphics-tools audience, worth a compat note in the README.

### vs native OIDN

- **Quality: parity.** Our ONNX conversion matches Intel's native output to
  ~54 dB (CPU-EP proof chain in `tools/oidn-native-compare/`); the shipped
  WebGPU path with splitAux is within 1–2 LSB. Same weights (verified
  byte-identical with upstream latest).
- **Speed: beats native CPU** (334 ms), **~4× off native Metal** (24.5 ms) at
  the same task. The gap is hardware matrix units (`simdgroup_matrix`), which
  WGSL cannot reach today — measured, not guessed (`kernel-spike`, NO-GO at
  ≤1.32×). Closes when WebGPU subgroup-matrix ships or via WebNN/NPU.
- **Features: subset.** We ship the RT (interactive/final-frame) models. Native
  OIDN also has RTLightmap models (not yet converted — cheap to add, converter
  is generic) and (future, OIDN 3.x) temporal models.

## Use cases (and whether we serve them today)

| Use case | Works? | Example today | Gap |
|---|---|---|---|
| three.js WebGPU path tracing (progressive) | ✅ flagship; zero-copy, shared device | `three-pathtracer-webgpu`, `ldraw-eiffel` | pathtracer dep is an unreleased branch (SHA-pinned) |
| Denoise a static render (image/canvas → image) | ✅ simplest API path | **none** — only verification harnesses | need a 10-line "hello world" example + before/after gallery |
| In-pipeline for ANY WebGPU renderer (Babylon, wgpu/WASM, custom) | ✅ engine-agnostic: textures/buffers/ImageData in-out | none | need one non-three example to prove neutrality |
| Offline-render preview in the browser (Blender/Cycles low-spp + AOVs) | ✅ works via aux inputs | none | EXR/HDR loading recipe + demo assets |
| Lightmap baking denoise | ⚠️ model not converted yet | none | convert `rt_lightmap_*` TZAs; demo is a separate effort |
| Photo/sensor noise | ❌ off-label — weights trained on Monte-Carlo render noise | n/a | document as a non-goal (it "does something" but isn't designed for it) |
| Non-WebGPU browsers | ❌ by design | n/a | point to v1 / document requirement |

Current `/examples` are mostly **verification harnesses** (bench,
aux-split-verify, webgpu-ort-smoke, package-test), not user-facing demos. The
example gap is the biggest launch blocker — the library is more capable than
the examples show.

## Are we at the ceiling of current tech?

Mostly yes. Whole-frame inference, fp16, zero-copy IO, and WGSL pre/post are
all landed; the remaining ~4× vs native Metal is hardware access, not our
orchestration (measured in `kernel-spike`). Blocked-on-upstream, re-test per
release: graphCapture (ORT crash), the >3ch first-Conv bug (splitAux
workaround), WebGPU subgroup-matrix, WebNN behind flag. Still-ours: OIDN 2.x
engine-side ideas (tiling/scheduling) — worth one investigation pass, likely
minor.

## Phased plan

### Phase A — stabilize & release (now)
1. File the onnxruntime issue (repo ready:
   [DennisSmolek/ort-web-webgpu-conv-bug](https://github.com/DennisSmolek/ort-web-webgpu-conv-bug)).
2. CI minimum: build + `tsc` + converter parity (STATUS checklist).
3. Robustness: `denoiseTextures` warns/throws on silently-dropped aux;
   WebGPU-unavailable error message points at requirements/v1.
4. `denoiser-react`: decide port-or-drop (leaning drop-for-now: v2 API is
   ~2 calls, a wrapper adds little; revisit on demand).
5. Publish `denoiser@2.0.0`.

### Phase B — examples & docs site (the launch)
1. **Hello-world example**: img → denoise → canvas, ~10 lines, no three.js.
2. **Before/after gallery**: precomputed low-spp renders (Blender Cycles +
   albedo/normal AOVs), comparison slider, spp ladder (1/2/4/8). Sexiest thing
   we can ship **without** the pathtracer, and it shows pure quality.
3. Polish + deploy `three-pathtracer-webgpu` and `ldraw-eiffel` (interactive;
   pinned SHA is fine for a demo).
4. One non-three integration example (raw WebGPU or Babylon) to prove
   engine-neutrality.
5. **Docs site** in the pmndrs style (docs.pmnd.rs system builds from the
   `docs/` markdown tree): audit README + docs for v2 accuracy, restructure to
   the pmndrs docs conventions, GH-Pages-host the built examples and link/embed
   them. Include a WebGPU-support matrix page.
6. Launch: share for feedback (pmndrs discord, three.js forum, X).

### Phase C — expand (post-launch, demand-driven)
- **Lightmap track**: convert `rt_lightmap_hdr`/`rt_lightmap_dir` to
  models repo (cheap, do early); the actual three.js lightmapper integration
  is a **separate example/side repo** — it drags in a whole baking stack and
  shouldn't gate or bloat the core.
- **WebNN**: plan already sketched in [perf-plan.md](perf-plan.md) Phase 6
  (same ONNX files, `onnxruntime-web/all` bundle, `denoiser/webnn` subpath,
  CPU-tensor IO first since ORT 1.27 lacks MLTensor interop). Implementation
  stays deferred until WebNN ships unflagged / NPU targets matter. No new
  doc needed — it's written.
- OIDN 2.x engine-side investigation (weights identical; gains were engine
  choices — one pass to see what's portable).
- OIDN 3.x temporal models when released (H2 2026) — see
  [upscale-roadmap.md](../specs/upscale-roadmap.md).

### When the upstream ORT bug is fixed
- Re-verify with `aux-split-verify` against the fixed ORT; gate `splitAux`
  default by ORT version (fixed → plain model; older → split). Keep the
  `models-v2` tail artifacts hosted indefinitely (immutable-tag policy, and
  older ORT versions still need them).
- Each ORT release, also re-test graphCapture (perf upside was ~0 last time,
  but capture may matter more post-fix) and re-run `bench` for free wins.

## Non-goals (documented so we stop re-litigating)
- Photo denoising (wrong training domain).
- WebGL fallback (v1 exists; WebGPU-only is the point of v2).
- Custom WGSL conv engine (measured NO-GO until subgroup-matrix).
