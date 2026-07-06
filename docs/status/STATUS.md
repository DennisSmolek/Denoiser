# Project status & next actions

_Last updated: 2026-07-06 (branch `perf-v2`, pre-org-migration checkpoint)._

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

## Org-migration checklist (do these when moving the repo)

- [ ] Update `packages/denoiser/package.json` `homepage` + `bugs` (currently
      `github.com/dennissmolek/denoiser`) and `packages/denoiser-react` if kept.
- [ ] **Models hosting: publish a separate models npm package** (e.g.
      `@org/denoiser-models`) and point the default in
      `packages/denoiser/src/weights.ts` at its jsDelivr URL, version-pinned.
      Numbers: 46 files, 144 MB total (96 fp32 + 48 fp16); largest single file
      14.7 MB (jsDelivr per-file limit is 50 MB — all fine); consumers only
      fetch the one model they use (0.6–15 MB), never the set. Rationale &
      alternatives (GitHub Releases works; jsDelivr's `gh/` endpoint doesn't
      resolve LFS; OIDN upstream has no ONNX) are documented in the package
      README's "Models / weights". The org CDN stays out of defaults (not for
      third-party prod). Until published, the shipped default 404s — README
      says self-host.
- [ ] Git-LFS: `packages/denoiser/tzas/*.tza` are LFS-tracked — make sure the
      org repo has LFS enabled before pushing, and CI runners install `git-lfs`.
- [ ] The `webgpu-pathtracer` dependency is a **git branch**
      (`github:gkjohnson/three-gpu-pathtracer#webgpu-pathtracer`) — pin to a
      commit SHA before org CI depends on it, and watch for its npm release.
- [ ] CI: no workflows exist yet. Minimum useful set: `yarn build` +
      `tsc --noEmit` across workspaces, and the Python converter's
      `verify_parity.py` (pure CPU, runs headless).
- [x] ~~`/examples` gitignore~~ — fixed: the ignore line is removed; example
      sources are tracked normally (node_modules/dist stay ignored globally,
      harness `dumps/` dirs ignored explicitly).
- [ ] Branches: `feat-v2` (migration) and `perf-v2` (perf + v2 API, current) —
      squash-merge or fast-forward into the org's `main`.
- [ ] Memory/handoff: this docs tree is the source of truth; the maintainer's
      local Claude memory also points here.

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
