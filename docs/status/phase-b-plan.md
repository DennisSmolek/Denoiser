# Phase B plan — examples, docs site, launch

_Last updated: 2026-07-10. The detailed execution plan for
[ROADMAP.md](ROADMAP.md) Phase B. Phase B gates the `denoiser@2.0.0` npm
publish. FSR3 demos are sequenced last (the companion lib is being finalized
in parallel)._

> **STATUS 2026-07-10 (final): PHASE B COMPLETE — ALL 10 demos + docs live.**
> Demo #8 (upscale-pipeline, `@pmndrs/upscaler`) landed same day the lib
> shipped; ONNX issue filed (microsoft/onnxruntime#29651); backlog moved to
> GitHub issues pmndrs/denoiser#1–12. Launch = issue #12 (redirect PR, npm
> publish, announce). Earlier same-day snapshot below:
>
> Live: 9 demos at
> https://pmndrs.github.io/denoiser/ (hello-world, gallery, aux-inputs,
> webgpu-raw, babylon, realtime-compare, bench, ldraw-eiffel,
> three-pathtracer-webgpu — all with screenshots) + the 14-page docs site at
> /docs/ (pmndrs docs image run directly; the reusable workflow can't build a
> nested mdx tree — see pages.yml comment). Remaining, all Dennis-gated:
> FSR3 demo (#5, companion lib), docs.pmnd.rs redirect PR, npm publish
> (B4.3 = the launch), ONNX issue filing. Deferred small items live in the
> session todo/STATUS: Inspector-overhead spike, eslint config wiring,
> upstream mdx-nesting bug report to pmndrs/docs.

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
   **Controls — three-based demos use the new three.js Inspector, not
   lil-gui** (`three/addons/inspector/Inspector.js`, shipped in our pinned
   r185; reference: three's `webgpu_postprocessing_ao.html`):
   - `renderer.inspector = new Inspector()`;
     `inspector.createParameters('Section')` gives lil-gui-style sections.
   - `.toInspector('Label')` on TSL texture/pass nodes = a free pass viewer —
     directly useful for showing color/albedo/normal aux inputs and the
     denoised pass (the `aux-inputs` demo can lean on this heavily).
   - **Overhead is an open question — spike before adopting everywhere**:
     bench a demo with (a) no inspector, (b) attached but closed, (c) open.
     If attached-closed isn't free, mount it behind `?inspector=1` and keep
     the default page clean. (It's also another internal-ish surface —
     re-verify per three release like the backend interop.)
   - Non-three demos (hello-world, webgpu-raw, babylon) can't use it; they
     keep the dependency-free chrome only.
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

### 5. `babylon` — Babylon.js integration (M)
Purpose: prove engine-neutrality, direction 2. **Spike resolved (2026-07-10):
no zero-copy** — Babylon's `WebGPUEngine` (current 9.16) cannot adopt an
external `GPUDevice` (`WebGPUEngineOptions` has only `deviceDescriptor`;
`initAsync` hard-codes adapter/device acquisition), and ORT can't adopt
Babylon's either (onnxruntime #26107). So this demo ships the **CPU-copy
path** and documents the boundary honestly: render to `RenderTargetTexture` →
`await rtt.readPixels()` (full-float, width divisible by 64 for the fast
path) → `denoise()` → upload via `RawTexture.update`. Scene: something
Babylon-idiomatic (PBR showcase style) with simple accumulated noise, not a
full path tracer.

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
**Spike resolved (2026-07-10):** available since exactly r185
(`three/addons/tsl/display/RecurrentDenoiseNode.js`, `recurrentDenoise()`;
canonical example `webgpu_postprocessing_ssr_denoise.html`). Design
consequence: it is a **screen-space-effect denoiser** ('diffuse' for AO/SSGI,
'specular' for SSR) needing depth/normal/metalRoughness/raw G-buffer inputs —
not a beauty-pass denoiser. So the fair comparison is an SSGI/SSR-style noisy
effect denoised both ways, NOT a path-traced beauty frame (where it isn't
designed to compete). Frame the page accordingly.

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
2. **Publish via the pmndrs docs system** — spike resolved (2026-07-10):
   fully self-service. Add `.github/workflows/docs.yml` calling the reusable
   workflow `pmndrs/docs/.github/workflows/build.yml@v3` (inputs: `mdx:
   'docs'`, `libname`, `home_redirect`, icon/logo) + `actions/deploy-pages`;
   enable Pages (source: GitHub Actions) → live at
   `pmndrs.github.io/denoiser`. Docs must be **.mdx** with `title` /
   `description` / global-numeric `nav` frontmatter; folder name = nav
   section label; images served from the raw branch URL; `<Codesandbox
   embed>` / `<Sandpack>` give live embeds. The friendly URL is a separate,
   later step: PR a redirect into pmndrs/docs `next.config.mjs`
   (`docs.pmnd.rs/denoiser/*` → Pages), or org-admin DNS for
   `denoiser.docs.pmnd.rs`.
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

Open questions — three of four resolved 2026-07-10 (results inline above:
pmndrs docs = self-service reusable workflow; Babylon = no shared device,
CPU-copy path; RecurrentDenoiseNode = in r185, screen-space effects only).
Still open: three.js Inspector overhead (none / attached-closed / open —
decide default-on vs `?inspector=1`).
