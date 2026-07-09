# Early optimization advice notes

> **ARCHIVED.** These suggestions were executed (or rejected with measurements)
> in `docs/status/perf-plan.md` — see its Results section.

One probable bottleneck in your current loop is that every 256 tile is submitted as extract → await ORT run → accumulate, which gives you many tiny GPU/JS synchronization points. That does not mean the model is wrong, but it does point to batching/pipeline changes as higher-impact than micro-optimizing the WGSL.

Current docs line up with what the code is already doing well: preferredOutputLocation: 'gpu-buffer' and Tensor.fromGpuBuffer() are the right direction. The bigger remaining question is whether ORT can reduce the repeated WebGPU dispatch overhead via graph capture/batching, and whether the converted model is actually fp16 end-to-end rather than fp16 weights with hidden casts.


Without WebNN, the honest ceiling is lower: the web cannot directly use Apple Neural Engine, Intel AMX-FP16, XMX, CUDA, Metal Performance Shaders, etc. Native OIDN is fast because it can use those lower-level kernels. The best browser path is squeezing WebGPU hard.

For this repo specifically, the highest-impact options look like:

Batch tiles instead of running one tile at a time.
Your engine currently does extract tile → await session.run() → accumulate for every 256 tile. That creates many JS/GPU scheduling points. If the ONNX model can be exported as [B, C, 256, 256], you could extract 4/8/16 tiles into one input buffer, run once, then accumulate all outputs. This is probably the biggest architectural speed win.

Try ORT WebGPU graph capture.
Your fixed tile shape is a good fit. enableGraphCapture: true can reduce CPU-side command preparation after the first run, assuming every op stays on WebGPU and shapes do not change.

Make fp16 actually end-to-end.
The package has fp16 model support, but the engine currently creates float32 GPU tensors. If the model inputs/outputs are still fp32, ORT may insert casts and lose much of the benefit. Better target: fp16 ONNX weights, fp16 model IO, Tensor.fromGpuBuffer(... dataType: 'float16'), WGSL f16 tile buffers, then accumulate/resolve in f32 only where needed.

Increase tile size or reduce overlap where quality allows.
A 256 tile with 32 overlap means lots of duplicate inference. Larger fully-convolutional tiles, multiple exported fixed sizes, or a “fast preview” overlap setting could reduce tile count dramatically.

Keep render data on the GPU.
If this is fed from three/WebGPU, avoid getImageData / CPU RGBA paths. Render color/albedo/normal into WebGPU textures or buffers, convert directly to model input, and only read back when the user needs an actual image file.

Profile for CPU fallback.
One unsupported ONNX op can quietly destroy performance by forcing CPU/WASM copies. ORT WebGPU profiling/logging is worth doing before deeper optimization.

I would not expect a WASM build of native OIDN to match native. WASM SIMD/threads can help, but it still cannot access AMX-FP16, AVX-512 width, XMX, Neural Engine, or native GPU inference stacks. Good fallback, not the main speed path.

The likely best path is: graph capture first, then batched tiles, then true fp16 IO, then larger-tile experiments.