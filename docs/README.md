# Documentation

## Guides (start here)

- [**three.js render targets in / out**](guides/three-js-render-targets.md) —
  the zero-copy integration path: device sharing, unwrapping render targets to
  `GPUTexture`, the full parameter reference, output-into-three, live-loop
  pacing, pitfalls.
- [**Migrating from v1 (TFJS)**](guides/migrating-from-v1.md) — the 0.x → 2.x
  API mapping, requirement changes, weights hosting.

The package README ([`packages/denoiser/README.md`](../packages/denoiser/README.md))
covers install, quick start, the API sketch, and the device-lifetime rules.

## Status / planning (`status/`)

- [**STATUS.md**](status/STATUS.md) — current state, **next actions**, known
  issues, and the org-migration checklist. The living doc.
- [MIGRATION_PROGRESS.md](status/MIGRATION_PROGRESS.md) — historical record of
  the TFJS → WebGPU/ONNX migration (architecture, gotchas, commit map).
- [perf-plan.md](status/perf-plan.md) — the performance work: plan, executed
  phases, measured results (bench harness, fp16, whole-frame, zero-copy…).
- [speedup.md](status/speedup.md) — early optimization advice notes (since
  executed; kept for context).

## Specs / references (`specs/`)

- [api-v2-spec.md](specs/api-v2-spec.md) — the v2 stateless-API design (implemented).
- [oidn-color-reference.md](specs/oidn-color-reference.md) — OIDN's color
  pipeline (PU transfer, autoexposure, normal encoding) and how ours matches it.
- [wgsl-engine-proposal.md](specs/wgsl-engine-proposal.md) — bespoke WGSL
  inference engine side-project (proposal; measured ceiling in
  [`tools/oidn-native-compare`](../tools/oidn-native-compare/README.md)).
- [upscale-roadmap.md](specs/upscale-roadmap.md) — denoise→upscale (FSR1 done,
  temporal later).
