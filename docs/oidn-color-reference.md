# OIDN upstream color pipeline — exact reference (fetched from RenderKit/oidn master)

Implementation reference for matching native OIDN's input/output processing.
Sources: core/color.h, core/color.cpp, core/autoexposure.h,
devices/cpu/cpu_autoexposure.{cpp,ispc}, core/rt_filter.cpp,
core/unet_filter.cpp, devices/gpu/gpu_{input,output}_process.h.

## PU transfer function (HDR color)

Constants (f32):
```
a=1.41283765e+03  b=1.64593172e+00  c=4.31384981e-01  d=-2.94139609e-03
e=1.92653254e-01  f=6.26026094e-03  g=9.98620152e-01
y0=1.57945760e-06 y1=3.22087631e-02      // forward breakpoints (linear)
x0=2.23151711e-03 x1=3.70974749e-01      // inverse breakpoints (encoded)
```
```
pu_forward(y) = y<=y0 ? a*y : y<=y1 ? b*pow(y,c)+d : e*log(y+f)+g   // ln
pu_inverse(x) = x<=x0 ? x/a : x<=x1 ? pow((x-d)/b, 1/c) : exp((x-g)/e)-f
```
Normalization: `yMax = 65504` (HDR_Y_MAX); `xMax = pu_forward(yMax)` (≈3.1355689,
verify in f32); `normScale = 1/xMax`.
Full HDR forward: `x = pu_forward(y) * normScale`; inverse: `y = pu_inverse(x * xMax)`.

## Which curve when (rt_filter.cpp newTransferFunc)

```
srgb flag set (input already display-encoded) → Linear (identity)
hdr                                           → PU (+ normScale)
else (linear LDR input)                       → SRGB curve forward
```
NOTE vs our lib: our RGBA8 default path feeds display-encoded bytes raw = matches
upstream "srgb→Linear". Our `srgb:true` option DECODES to linear before the net —
backwards vs upstream (they'd ENCODE a linear input). v2 should redefine: flag
means "input is linear → apply sRGB forward pre-net, inverse post-net".

## Autoexposure (HDR inputScale)

maxBinSize=16, key=0.18, eps=1e-8. On the RAW hdr color (before scale/transfer):
- Grid numBins = ceil(H/16) × ceil(W/16); integer bin bounds i*H/numBinsH.
- Per pixel: c = clamp(nan→0, 0, FLT_MAX); L = 0.212671r + 0.715160g + 0.072169b.
- Per bin: mean L. Over bins with L>eps: accumulate log2(L), count.
- inputScale = count>0 ? 0.18 / exp2(sumLog2/count) : 1.0. outputScale = 1/inputScale.
- User-provided inputScale overrides; LDR → 1.0.

## Full pixel pipelines (gpu_input_process.h / gpu_output_process.h)

Input color: `v *= inputScale` → `clamp(nan→0, 0, hdr?FLT_MAX:1)` → `transfer.forward(v)`.
Output color: `clamp(nan→0, 0, FLT_MAX)` → `transfer.inverse(v)` → if !hdr `min(v,1)` → `v *= outputScale`.

## Aux channels — IMPORTANT DIFFERENCES FROM OUR CURRENT CODE

- Albedo: `clamp(nan→0, 0, 1)` only. (matches us)
- **Normal: `clamp(nan→0, -1, 1)` then `v*0.5+0.5` → network sees [0,1]!**
  We currently feed [-1,1] (RGBA8 path does *2-1; texture path passes raw).
  FIX: RGBA8 path feed bytes raw [0,1]; texture path apply *0.5+0.5.
- Channel order: color 0-2, albedo 3-5, normal 6-8 (matches us).

## Also queued with this work

- Converter default `--final-activation none` (upstream output conv has no
  activation; relu6 was old-TF parity) + regenerate all models.
