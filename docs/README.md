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
- [ROADMAP.md](status/ROADMAP.md) — strategic view: v1/native-OIDN comparison,
  use-case coverage, and the phased plan to the public v2 launch.
- [MIGRATION_PROGRESS.md](status/MIGRATION_PROGRESS.md) — historical record of
  the TFJS → WebGPU/ONNX migration (architecture, gotchas, commit map).
- [perf-plan.md](status/perf-plan.md) — the performance work: plan, executed
  phases, measured results (bench harness, fp16, whole-frame, zero-copy…).

## Specs / references (`specs/`)

- [oidn-color-reference.md](specs/oidn-color-reference.md) — OIDN's color
  pipeline (PU transfer, autoexposure, normal encoding) and how ours matches it.
- [upscale-roadmap.md](specs/upscale-roadmap.md) — denoise→upscale (FSR1 done,
  temporal later).

## Archive (`archive/`)

Superseded proposals and notes (implemented specs, NO-GO experiments) — see
[archive/README.md](archive/README.md).
