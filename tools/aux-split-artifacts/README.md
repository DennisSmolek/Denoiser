# aux-split-artifacts — split-graph artifacts for the aux WebGPU workaround

Generates the per-model artifacts that make the 9-channel cleanAux denoisers work
on onnxruntime-web's WebGPU EP, which otherwise miscomputes the first conv that
reduces the raw >3-channel input (see `../ort-webgpu-aux-repro` for the bug,
`../ort-webgpu-aux-split` for the isolation). At runtime the denoiser computes
that first conv in a WGSL kernel and runs the rest (the "tail") on ORT-WebGPU.

For **every** cleanAux model (`rt_{hdr,ldr}_calb_cnrm{,_small,_large}` × fp32/fp16)
it emits two files next to the model:

- `<name>[.fp16].tail.onnx` — the network from the second conv on; two inputs
  `[<first-conv-relu6-output>, input]` (the input skip still needs the raw image).
- `<name>[.fp16].enc0.bin` — the first conv's weights, **f32** OIHW
  `[COUT, 9, 3, 3]` then bias `[COUT]` (f32 even for fp16 models; the kernel
  accumulates in f32). `COUT` is 32 for base/small, 64 for large.

The first conv is found generically (base/small: `enc_conv0`; large: `enc_conv1a`).

## Generate

```sh
tools/onnx-convert/.venv/bin/python tools/aux-split-artifacts/generate.py
# -> tools/aux-split-artifacts/out/  (20 files, ~36 MB; gitignored)
```

Each split is CPU-verified faithful (`full(x) == head(x)+tail(...)`), so the
input skip is never dropped. Real-engine WebGPU verification (fp32 + fp16, split
output byte-matches the native reference while the plain model speckles) lives in
`examples/aux-split-verify`.

## Host (this is the step that turns aux on for everyone)

The runtime fetches `<weightsUrl>/<name>[.fp16].tail.onnx` and `…​.enc0.bin` —
i.e. the artifacts must sit **next to the models** in the `pmndrs/denoiser-weights`
repo. `splitAux` is **on by default**; until these are hosted it falls back to the
plain (speckled) model with a one-time console warning.

1. Copy `out/*` into `denoiser-weights/models/` (alongside the `.onnx` files).
2. Commit and **re-tag** so the default CDN URL resolves them:
   - the default `weightsUrl` is `…/gh/pmndrs/denoiser-weights@models-v1/models`.
   - Either force-move the `models-v1` tag to the new commit, **or** create
     `models-v2` and bump the default in `packages/denoiser/src/weights.ts`
     (`Models.url`).
3. Verify a fetch: `curl -I <weightsUrl>/rt_hdr_calb_cnrm_small.tail.onnx` → 200,
   CORS `*`. jsDelivr gh serves plain git blobs with range + CORS (see the
   weights repo README; do not use GitHub Releases — no CORS).

That's it — no code change needed beyond the optional tag bump. `splitAux` picks
the right file per precision automatically.
