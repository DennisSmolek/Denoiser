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

1. **Aux speckle bug — ours, confirmed, open.** Native OIDN denoises our exact
   dumped inputs perfectly clean (`tools/oidn-native-compare`, results in its
   README), so the residual speckle in the web aux path is our bug. Next step:
   diff our aux output against `native_aux.pfm` on the same inputs, then bisect
   the WGSL color pipeline (PU/autoexposure/normal encode order —
   `docs/specs/oidn-color-reference.md`).
   _2026-07-07 progress (example-side, in `examples/ldraw-eiffel`):_ fixed three
   aux-generation bugs — env-background normals were a garbage gradient (OIDN
   wants normal=0 for env; G-buffer is now split albedo/normal passes),
   transparent surfaces alpha-blended their normals (normal pass now forces
   first-hit), and albedo could exceed [0,1] from HDR env colors (engine now
   clamps, `ort/wgsl.ts`). That example also has input-view debug tooling
   (color/albedo/normal exactly as the network sees them) — use it for the
   engine-side bisect that remains.
2. **WGSL engine spike** (`docs/specs/wgsl-engine-proposal.md`): one fused
   conv3×3+relu6 tiled kernel benchmarked against ORT's conv on the dominant
   shapes. Go/no-go gate: ≥1.4×. Only after (or parallel to) the aux fix.
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
