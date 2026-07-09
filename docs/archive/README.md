# Archive

Historical docs kept for the record — nothing here describes current behavior.

- [api-v2-spec.md](api-v2-spec.md) — the v2 stateless-API proposal. **Implemented**
  (2.x is this API); current docs live in the package README and
  [`guides/three-js-render-targets.md`](../guides/three-js-render-targets.md).
- [wgsl-engine-proposal.md](wgsl-engine-proposal.md) — bespoke WGSL inference
  engine proposal. **Measured NO-GO** (best kernel 1.32×, under the 1.4× gate;
  results in the `kernel-spike` branch's `experiments/wgsl-conv/SPIKE.md` and
  [`status/STATUS.md`](../status/STATUS.md)). Revisit when WebGPU
  subgroup-matrix ships.
- [speedup.md](speedup.md) — early optimization advice notes. Superseded by
  [`status/perf-plan.md`](../status/perf-plan.md) (the executed plan + measured
  results).
