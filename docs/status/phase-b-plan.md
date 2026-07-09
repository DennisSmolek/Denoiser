# Phase B plan — examples, docs site, launch

_Last updated: 2026-07-09. The detailed execution plan for
[ROADMAP.md](ROADMAP.md) Phase B. Phase B gates the `denoiser@2.0.0` npm
publish. FSR3 demos are sequenced last (the companion lib is being finalized
in parallel)._

## Shape of the work

Five stages. B0–B1 are prerequisites everything else builds on; B2 demos land
incrementally (each is independently shippable); B3 docs content can proceed in
parallel with B2; B4 wires it all up and launches.

```
B0 content audit ─┬─ B2 demos (1→7, FSR last) ─┬─ B4 deploy + launch
B1 demo infra ────┘   B3 docs site content ────┘
```

---

## B0 — Content audit & docs restructure

Goal: every claim in README/guides is true for 2.0, and `docs/` is shaped so
the pmndrs docs system can consume it.

1. Accuracy pass over root README, `packages/denoiser/README.md`,
   `guides/three-js-render-targets.md`, `guides/migrating-from-v1.md` —
   verify every code sample against the shipped API (run them, don't read them).
2. **Verify current pmndrs/docs conventions first** (the system that serves
   docs.pmnd.rs — MDX + frontmatter out of the repo's `docs/` tree; check how
   r3f/drei structure theirs today, nav ordering, image handling, code
   embeds). Restructure to match.
3. Split "site docs" from "engineering docs": the site nav shows
   getting-started/concepts/guides/API/examples only; `status/`, `specs/`,
   `archive/` stay repo-only (excluded from nav, still linked from a
   "contributing/internals" page).
4. WebGPU support matrix page content (Chrome/Edge stable, Safari 26+,
   Firefox status; the max-limits requirement; "v1 for WebGL" note).

## B1 — Demo infrastructure (build once, every demo uses it)

1. **Example harness**: shared vite config so each `examples/*` builds to a
   static bundle under a common base path; one GitHub Actions workflow builds
   all examples + deploys to GitHub Pages on push to main.
2. **Shared demo chrome**: tiny common package/snippet — WebGPU feature
   detect with a friendly "needs WebGPU" banner (link to support matrix),
   FPS/ms overlay, standard footer (repo/docs links). Keep it dependency-free.
3. **Demo assets**: noisy renders + AOV ladders for the gallery demos.
   Primary plan: **dogfood** — a capture mode in the existing pathtracer
   examples that saves frames at 1/2/4/8/16 spp plus albedo/normal AOVs
   (EXR or 16f PNG). Optional garnish: 1–2 Blender classic scenes (CC-licensed
   demo files) rendered in Cycles for visual variety. Host small PNGs in-repo
   (they deploy with Pages); anything heavy goes in a `demo-assets` tag on the
   weights repo (jsDelivr, same policy as models).

## B2 — The demos

Ordered for momentum: quick wins first, riskier integrations later, FSR last.

### 1. `hello-world` (S)
Purpose: the 10-line story. Noisy image → `Denoiser.create()` → `denoise()` →
canvas. No three.js, no build tricks. The code IS the marketing.
Page shows the source alongside the result. This becomes the docs
"Getting started" embed.

### 2. `gallery` — before/after + spp ladder (M) — **flagship quality demo**
- Comparison slider (noisy | denoised) over 3–5 scenes.
- spp ladder selector (1/2/4/8/16) — shows quality vs input noise.
- Aux toggle: color-only vs +albedo+normal, same frame — *this is where aux
  earns its complexity*, and it doubles as the public proof the splitAux fix
  works.
- Runs live inference on precomputed inputs (assets from B1.3), so it shows
  real in-browser speed without needing the pathtracer.

### 3. `aux-inputs` — what the network sees (S/M)
Purpose: education + debugging reference. Show color/albedo/normal exactly as
fed to the network (seeded by the ldraw-eiffel input-view debug tooling),
with correct/incorrect examples (env normals, alpha-blended normals, HDR
albedo clamping — the real mistakes we made). Doubles as the "aux inputs"
guide's live companion.

### 4. `webgpu-raw` — no framework (M)
Purpose: prove engine-neutrality, direction 1. A single-file WGSL toy path
tracer (sphere/box scene, ~200 lines, progressive accumulation) →
`denoiseTextures` on the same device → canvas. No three, no Babylon, no
bundler magic. Also serves as the reference for "integrate with a custom
engine" docs.

### 5. `babylon` — Babylon.js integration (M, **risk item**)
Purpose: prove engine-neutrality, direction 2. **Open question to resolve
first** (30-min spike): can Babylon's `WebGPUEngine` adopt an external
`GPUDevice` (ours), the way three's `WebGPURenderer({ device })` can? If yes:
zero-copy path, same story as three. If no: ship the CPU-copy path
(readback → `denoise` → texture upload) and say so honestly — still a working
integration, and it documents the boundary. Scene: something Babylon-idiomatic
(their PBR showcase style) with simple accumulated noise (jittered AO or
env-sampling), not a full path tracer.

### 6. Polish + deploy existing pathtracer demos (S/M)
`three-pathtracer-webgpu` + `ldraw-eiffel`: demo chrome, sane defaults,
pinned-SHA note ("pathtracer is pre-release upstream"), deploy. These are the
"wow" interactive pieces until the pathtracer releases properly.

### 7. `realtime-compare` — vs three.js `RecurrentDenoiseNode` (M)
Purpose: honest positioning. Same scene (three WebGPURenderer), split view:
- Left: RecurrentDenoiseNode (temporal, lightweight, every frame).
- Right: ours (progressive refinement on accumulation / on-demand final frame).
- Live ms-per-frame + quality readout (PSNR vs converged reference where
  possible).
Narrative on the page: different points on the quality/latency curve —
temporal realtime for motion, OIDN-class for stills/converged frames;
complementary, not competing. (Temporal OIDN waits for 3.x weights — say so.)
**Verify first**: RecurrentDenoiseNode's API/availability in r185+ (it's
recent; check exact import and requirements).

### 8. `fsr3-pipeline` — denoise → upscale (M, **LAST**, blocked on companion lib)
Pathtrace/render at 512–720p → denoise (hdr) → FSR3 upscale to canvas res,
all on one shared device. The "three packages, one GPUDevice, zero copies"
story. Optionally retrofit the gallery with an upscale toggle. Supersedes the
FSR1 stage in the pathtracer demo; update
[upscale-roadmap.md](../specs/upscale-roadmap.md) when it lands.

## B3 — Docs site content (parallel with B2)

Proposed information architecture (pmndrs-docs style, one MDX per page):

**Getting started**
1. *Introduction* — what it is (OIDN in the browser, full WebGPU), hero
   before/after embed, when to use it (and when not: photos, realtime),
   browser support matrix.
2. *Quick start* — install, hello-world (live embed), the three lines that
   matter, models auto-fetch from CDN.
3. *Models & weights* — quality tiers (fast/balanced/high), LDR vs HDR,
   what aux buys, download sizes, self-hosting (`weightsUrl`), the
   jsDelivr/immutable-tags policy, offline/bundling notes.

**Concepts**
4. *How it works* — OIDN U-Net → ONNX → onnxruntime-web WebGPU EP, WGSL
   pre/post, whole-frame vs tiling (the receptive-field story), fp16.
5. *The GPUDevice* — who owns it (ORT), sharing with your renderer,
   max-limits patch, device lifetime rules, the one-denoiser-per-device rule.
6. *Performance* — measured numbers (bench tables), what affects speed
   (size/precision/quality/aux), progressive-denoise pacing patterns,
   vs native OIDN honestly.

**Guides**
7. *three.js render targets in/out* — exists, port + verify.
8. *Generating aux inputs* — G-buffer/MRT recipes, OIDN's expectations
   (normals for env hits, albedo clamping, first-hit rules), links to the
   aux-inputs demo.
9. *Custom engines / raw WebGPU* — textures & buffers in/out, formats,
   flipY, based on the `webgpu-raw` demo.
10. *Large images & tiling* — `maxRunPixels`, the tile ladder, memory math.
11. *Migrating from v1* — exists, port.
12. *Troubleshooting / FAQ* — no WebGPU, black output, upside-down output,
    device-lost, CORS/hosting, "why is my aux output speckled" (old ORT +
    splitAux off).

**Reference**
13. *API* — `Denoiser.create` options table, `denoise`, `denoiseTextures`,
    types, errors. Generated-or-handwritten TBD (handwritten first; the API
    is small).

**Examples**
14. *Examples index* — card grid linking every deployed demo with a
    screenshot; each demo page back-links its guide.

## B4 — Deploy, wire up, launch

1. **GitHub Pages** on `pmndrs/denoiser` from the B1 workflow (examples +
   any static assets). Custom domain optional and additive later.
2. **Register with the pmndrs docs system** — PR/config so docs.pmnd.rs
   serves the `docs/` tree (verify the current registration process against
   pmndrs/docs; r3f is the template).
3. **npm**: publish `2.0.0` (this is the launch trigger, not before).
4. Launch posts: pmndrs Discord, three.js forum, X. Lead with the gallery +
   a pathtracer demo link.

### Dennis's checklist (things only you can do)
- [ ] **npm**: confirm you still control the `denoiser` package name;
      repository/homepage fields already point at pmndrs (done in repo) but
      the *published* package metadata updates only on next publish. Decide
      whether org members should be added as npm maintainers.
- [ ] **Domain/DNS**: if we want a custom domain for the examples site
      (e.g. `denoiser.pmnd.rs` CNAME → `pmndrs.github.io`), coordinate the
      subdomain with the pmndrs admins; otherwise
      `pmndrs.github.io/denoiser/` works day one with zero DNS.
- [ ] **pmndrs/docs registration**: approval/merge on the org side.
- [ ] **GitHub Pages**: enable Pages on `pmndrs/denoiser` (Settings → Pages →
      GitHub Actions source) — needs repo admin.
- [ ] **FSR3 lib**: finalize + name/publish it (demos 8 blocked on it).
- [ ] Optional: pick/approve gallery scenes (I'll default to dogfooded
      captures + 1–2 CC Blender classics).

## Sequencing

| step | items | blocked by |
|---|---|---|
| 1 | B0 audit + B1 infra | — |
| 2 | Demos 1–3 (hello-world, gallery, aux-inputs) | B1 |
| 3 | B3 docs pages 1–6, 12–14 drafted | B0 |
| 4 | Demos 4–7 (raw WebGPU, Babylon spike→demo, pathtracer polish, realtime-compare) | B1 |
| 5 | B3 guides 7–11 finalized against the demos | step 4 |
| 6 | B4.1–B4.2 deploy (site live in "soft launch") | steps 2–5 |
| 7 | Demo 8 (FSR3) | companion lib ready |
| 8 | B4.3–B4.4 npm publish + announce | everything above |

Open questions to resolve early (cheap spikes, in order): pmndrs/docs current
registration process; Babylon external-device adoption;
RecurrentDenoiseNode exact API/availability in r185+.
