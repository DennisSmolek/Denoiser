# aux-split-verify

Real-engine verification of the aux split-graph workaround for the onnxruntime-web
WebGPU Conv bug (see `tools/ort-webgpu-aux-repro` + `tools/ort-webgpu-aux-split`).

It drives the actual `DenoiseEngine` in **split mode** — our WGSL `enc_conv0`
kernel + a re-exported tail model on ORT-WebGPU — and, as a baseline, in normal
mode (full model on ORT-WebGPU, which hits the bug). Both are compared to a WASM
full-model reference computed on the identical `bytes/255` input (the CPU
`denoise()` path with `srgb:false, hdr:false` makes the NCHW exactly `bytes/255`,
so the reference is reproducible).

## Result (onnxruntime-web 1.27.0, headless Chrome, ANGLE/Metal, M-series)

| path | max byte Δ vs ref | mean Δ | pixels >2 LSB | noise (R local-var) |
|---|---|---|---|---|
| full model on ORT-WebGPU (bug) | 27 | 3.11 | 48.9% | 93.1 |
| **split (WGSL enc_conv0 + tail)** | **1** | **0.000** | **0.00%** | 68.6 (= ref) |

The split path is byte-identical to the reference; the unmodified WebGPU path is
speckled. Confirms the workaround is correctly wired into the runtime and clears
the bug.

## Run

```sh
# 1. build the split artifacts (needs the onnx venv; models are gitignored)
python3 build_split.py                 # -> public/models/{full,tail}.onnx, enc0.bin
# 2. dev server (imports DenoiseEngine from ../../packages/denoiser/src)
yarn dev                               # open the printed URL in a WebGPU browser
```

`build_split.py` reads `packages/denoiser/models/rt_hdr_calb_cnrm.onnx`, extracts
the tail (`enc_conv1..output`, inputs `[enc_conv0_relu6_2, input]`), and dumps
`enc_conv0` weights as `enc0.bin` (f32 OIHW `[32,9,3,3]` then bias `[32]`).
