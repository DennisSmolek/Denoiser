# Split-graph test — does externalising `enc_conv0` clear the aux bug?

Companion to `../ort-webgpu-aux-repro` (which proved the OIDN aux speckle is a
bug in **onnxruntime-web's WebGPU execution provider**, not our port). This test
answers the *next* question, **before** we build the real fix: if we compute the
first conv (`enc_conv0`) ourselves and let ORT-WebGPU run everything else, does
the output actually come out clean?

It is not obvious that it would. The OIDN U-Net reduces the raw 9-channel `input`
tensor in **two** convs:

- `enc_conv0` — the first layer (`Conv 9 → 32`, 3×3), and
- `dec_conv1a` — fed by `concat_38 = concat(up_37, input)`, reducing `96 + 9 =
  105` channels, **including the same raw 9ch input**.

If the WebGPU bug were "any conv reducing the raw >3ch input", externalising only
`enc_conv0` would leave `dec_conv1a` still broken and the workaround would fail.

## Method

Split `rt_hdr_calb_cnrm.onnx` (base 9ch aux) at `enc_conv0_relu6_2`:

- `head_enc0.onnx` — `input(9)` → `enc_conv0_relu6_2(32)`  (Conv + relu6 only)
- `tail.onnx` — `enc_conv0_relu6_2(32)` **and** `input(9)` → `output(3)`  (the
  whole rest of the net, including the `dec_conv1a` raw-input skip)

WASM (CPU) is the correctness oracle (it matches native OIDN). Running `enc_conv0`
on WASM stands in for our WGSL kernel (which is correct to 2e-6). Then run the
tail on WASM (sanity) and on WebGPU (the real workaround), same fixed input.

## Result (onnxruntime-web 1.27.0, headless Chrome, ANGLE/Metal, M-series)

| pipeline | max\|Δ\| vs WASM ref | noise ratio | verdict |
|---|---|---|---|
| full model on WebGPU (baseline: the bug) | 1.08e-1 | 1.36× | ❌ diverges |
| head + tail both on WASM (split faithful?) | 0.0 | 1.00× | ✅ exact |
| **`enc_conv0` off-GPU + tail on WebGPU** | **1.19e-6** | **1.00×** | ✅ **matches** |

**The workaround works.** Externalising just `enc_conv0` restores full
native-quality output. `dec_conv1a` — a wider conv that also ingests the raw 9ch
input — computes correctly on WebGPU, so the fault is specifically the **model's
first Conv on the raw > 3-channel graph input**, nothing else.

## Run

```sh
# 1. generate the split models (gitignored — derived from the tracked base model)
../onnx-convert/.venv/bin/python make_split.py    # writes models/{full,head_enc0,tail}.onnx
# 2. serve + open
python3 -m http.server 5182                        # from this directory
# open http://localhost:5182 in a WebGPU browser; results print + land on
# window.__results
```

`make_split.py` extracts `head_enc0.onnx` + `tail.onnx` from
`packages/denoiser/models/rt_hdr_calb_cnrm.onnx` via `onnx.utils.extract_model`
and copies `full.onnx` for the baseline. `models/` is gitignored (reproducible).
ORT loads from the jsDelivr npm CDN.

## Why this matters for the product

This validates the shipping fix: **our WGSL computes `enc_conv0`, ORT-WebGPU runs
a re-exported tail starting at `enc_conv1`** (two inputs — the feature map and the
raw `input` for the skip). No CPU fallback, full quality. Extend to the other aux
variants (`_small`, `_large`, LDR, fp16) by re-exporting each the same way.
