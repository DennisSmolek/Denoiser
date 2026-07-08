# Project status & next actions

_Last updated: 2026-07-07 (branch `main`; `perf-v2`/`feat-v2` fully merged)._

The living "where are we" doc. History lives in [`MIGRATION_PROGRESS.md`](MIGRATION_PROGRESS.md)
(the TFJS→WebGPU/ONNX migration) and [`perf-plan.md`](perf-plan.md) (perf phases
+ measured results); this file is the forward-looking summary.

## Current state (one paragraph)

`denoiser@2.0.0-alpha.0`: OIDN U-Nets as ONNX on onnxruntime-web's WebGPU EP,
all pre/post as WGSL compute on one shared `GPUDevice`. Stateless v2 API
(`Denoiser.create` / `denoise` / `denoiseTextures`), zero-copy texture IO,
whole-frame inference to ~1080p with adaptive tiling above, fp16 end-to-end
(PSNR ≈ 53 dB vs fp32), MRT G-buffer aux, FSR1 2× upscale stage in the demo.
Measured warm (M-series, fast quality, fp16): 512² ≈ 14 ms, 720p ≈ 45 ms,
1080p ≈ 104 ms. We beat native OIDN CPU (334 ms); native Metal (24.5 ms)
defines a ~9× kernel-efficiency ceiling (~2–3× reachable via custom WGSL,
the rest locked behind hardware matrix units until WebGPU subgroup-matrix).

## Next actions (priority order)

1. **Aux speckle bug — CONFIRMED upstream 2026-07-08: onnxruntime-web WebGPU-EP,
   NOT our code.** Standalone repro in `tools/ort-webgpu-aux-repro/` (README =
   the upstream bug report): **bare vanilla ORT-web sessions** (no denoiser code,
   plain CPU-array IO), same fixed synthetic input, WebGPU vs WASM provider —
   3-ch model matches to 9.5e-7, but 6-ch diverges (3.7e-2) and 9-ch diverges
   more (1.1e-1). Error appears **above 3 input channels and scales with count**;
   fp32 + fp16 + base + small all affected. Points at ORT's WebGPU **Conv**
   first-layer input-channel reduction. (Earlier chain in
   `tools/oidn-native-compare/README.md` established our ONNX matches native to
   54 dB on CPU-EP.) **Resolution path:** (a) file upstream (onnxruntime; related
   #24070/#26734/#24442); (b) fp16 is NOT a workaround (tested); (c) best
   GPU-side fix — compute the one buggy first conv (`enc_conv0`) in our own WGSL
   (correct to 2e-6, see `kernel-spike`) and re-export the model to start at
   layer 1; (d) try newer/nightly ORT-web; (e) meanwhile keep aux off by default,
   mark experimental.
   _Also landed 2026-07-07 (example-side, `examples/ldraw-eiffel`):_ three
   aux-**generation** fixes independent of the above — split albedo/normal
   G-buffer passes (env normals were a garbage gradient; OIDN wants normal=0 for
   env), normal pass forces first-hit (no alpha-blended normals), and the engine
   clamps albedo to [0,1] (`ort/wgsl.ts`) for HDR env colors. Plus input-view
   debug tooling (color/albedo/normal as the network sees them). Robustness note
   found en route: `denoiseTextures` silently falls back to color-only when the
   aux textures resolve to undefined — should warn/throw.
2. **WGSL engine spike — DONE 2026-07-07: NO-GO.** Ran the full registry (9
   kernels: naive, 3 tiled-smem f32, 3 tiled-f16, 2 fused-relu6) against a fair
   gpu-buffer-IO ORT baseline on the 512² 64→64 3×3 shape (branch `kernel-spike`,
   `experiments/wgsl-conv/SPIKE.md`). Best custom kernel ≈1.32× (fused-relu6-f16,
   f16-precision); every **f32** tiled kernel ≤1.08× (parity with ORT). Under the
   ≥1.4× gate. Confirms the native-Metal gap is hardware matrix units
   (`simdgroup_matrix`), not orchestration — unreachable from WGSL until the
   subgroup-matrix proposal ships. Revisit then, or explore int8/DP4a with
   offline quantization. (1080p not measurable headless — ~2 GB of live buffers
   hangs the GPU; 512² decides the gate.)
3. **WebNN track** — deferred pending research (plan sketch in `perf-plan.md`).
   Re-evaluate when the WebNN EP matures / NPU targets matter.
4. **OIDN 2.x engine investigation** (user request): weights are byte-identical
   upstream, so OIDN 2.x's gains are engine-side — find what's worth porting
   (their tiling/IO/scheduling choices).
5. **Upscale roadmap** (`docs/specs/upscale-roadmap.md`): FSR1 is in the demo;
   evaluate temporal options later (blocked on OIDN 3.x temporal models H2 2026).
6. **Release engineering**: publish `2.0.0` (currently alpha) once the aux bug
   is fixed; changeset flow already in the repo.

## Org-migration checklist (executed 2026-07-06 → pmndrs)

- [x] **Migrated to `pmndrs/denoiser`.** All branches pushed (`main` = full
      current work, `perf-v2`, `feat-v2`, `kernel-spike`); LFS objects verified
      transferred (media URL serves real binaries). Personal repo
      (`DennisSmolek/Denoiser`) retains a full backup of every branch.
- [x] **Models hosted: `pmndrs/denoiser-weights`** — 46 `.onnx` (144 MB) as
      plain git blobs (no LFS), tag `models-v1`, default `weightsUrl` in
      `weights.ts` points at
      `https://cdn.jsdelivr.net/gh/pmndrs/denoiser-weights@models-v1/models`.
      **Verified live**: CORS `*`, range requests, byte-identical full
      downloads (jsDelivr transparently gzips fp32 models ~40%). Regeneration
      + retag policy documented in that repo's README. Verified dead ends
      (do not revisit): GitHub **Releases assets send no CORS headers**;
      npm bundling = install bloat + jsDelivr's ~50 MB total-package cap;
      raw.githubusercontent is dev-only. GitHub Pages is the verified backup.
- [x] `homepage`/`bugs` → `github.com/pmndrs/denoiser` in both packages.
- [x] Git-LFS verified on the org repo (tzas served as real content). CI
      runners will still need `git-lfs` installed.
- [ ] The `webgpu-pathtracer` dependency is a **git branch**
      (`github:gkjohnson/three-gpu-pathtracer#webgpu-pathtracer`) — pin to a
      commit SHA before org CI depends on it, and watch for its npm release.
- [ ] CI: no workflows exist yet. Minimum useful set: `yarn build` +
      `tsc --noEmit` across workspaces, and the Python converter's
      `verify_parity.py` (pure CPU, runs headless).
- [ ] `packages/denoiser-react`: still on the old v1 API — decide port or drop
      before publishing v2.
- [x] ~~`/examples` gitignore~~ — fixed: example sources tracked normally.
- [x] Memory/handoff: this docs tree is the source of truth.

## Known issues / limitations (beyond next-actions)

- **Aux speckle** (above) — the one known quality bug.
- `graphCapture` unusable in ORT 1.27 (crashes after ~150–250 replays);
  standalone repro exists (`ort-webgpu-graphcapture-repro`). Re-test each ORT
  release, keep opt-in until stable.
- WebGPU-only by design: no fallback; v1 remains the answer for non-WebGPU.
- three.js interop uses `renderer.backend.get(...)` (internal API) — re-verify
  per three release (guide: `docs/guides/three-js-render-targets.md`).
- Analytic lights unsupported by the WebGPU path tracer branch (environment
  lighting only) — upstream limitation, not ours.
