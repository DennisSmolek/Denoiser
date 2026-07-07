// Kernel spike harness: ORT-WebGPU = reference (correctness) + FAIR baseline
// (speed), then every registry kernel gets a max-abs-diff gate and a timed loop.
//
// FAIR BASELINE: ORT runs with gpu-buffer IO — Tensor.fromGpuBuffer feeds +
// preferredOutputLocation 'gpu-buffer', same pattern as packages/denoiser's
// ort/engine.ts — so no CPU<->GPU tensor copies land inside the timed loop.
// (The old scaffold used CPU-tensor IO, paying a 67MB readback/run that our
// kernels never pay; that inflated the naive speedup.)
//
// VRAM-frugal: at 1080p each full buffer is ~534MB. We share ONE input buffer
// and ONE f32 output buffer between ORT and our kernels (ORT's baseline runs
// fully before any kernel touches them) to keep peak VRAM bounded.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs';
import { kernels } from './kernels.js';

const SIZES = [{ w: 512, h: 512 }, { w: 1920, h: 1088 }];
const CIN = 64, COUT = 64, WARM = 3, ITERS = 12;
const el = document.getElementById('log');
const log = (m) => { el.textContent += m + '\n'; console.log(m); };
const results = { meta: { cin: CIN, cout: COUT, warm: WARM, iters: ITERS }, sizes: [] };
window.__results = results;
window.__done = false;

async function fetchF32(url) { return new Float32Array(await (await fetch(url)).arrayBuffer()); }

// f32 <-> f16 (IEEE half); use Float16Array when the runtime has it.
function f32ToF16Bits(src) {
  if (typeof Float16Array !== 'undefined') return new Uint16Array(new Float16Array(src).buffer);
  const out = new Uint16Array(src.length);
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < src.length; i++) {
    f32[0] = src[i];
    const x = u32[0], sign = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 112, m = (x >>> 13) & 0x3ff;
    if (e <= 0) { out[i] = sign; continue; }
    if (e >= 31) { out[i] = sign | 0x7c00; continue; }
    out[i] = sign | (e << 10) | m;
  }
  return out;
}
function f16BitsToF32(bits) {
  if (typeof Float16Array !== 'undefined') {
    return Float32Array.from(new Float16Array(bits.buffer, bits.byteOffset, bits.length));
  }
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i], sign = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) out[i] = sign * m * 2 ** -24;
    else if (e === 31) out[i] = m ? NaN : sign * Infinity;
    else out[i] = sign * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

async function readBuf(device, src, rb, bytes) {
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ, 0, bytes);
  const out = rb.getMappedRange(0, bytes).slice(0);
  rb.unmap();
  return out;
}

async function main() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
  const weights = await fetchF32('./weights.bin');
  const bias = await fetchF32('./bias.bin');
  const modelBytes = new Uint8Array(await (await fetch('./conv.onnx')).arrayBuffer());
  log(`conv ${CIN}->${COUT} 3x3 | warm ${WARM}, iters ${ITERS}`);

  const IO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  let device = null, hasF16 = false, mkBuf = null, errHooked = false;

  for (const { w, h } of SIZES) {
    log(`=== ${w}x${h} ===`);
    const sizeRes = { w, h, ort: {}, kernels: [] };
    results.sizes.push(sizeRes);
    const input = new Float32Array(CIN * w * h);
    for (let i = 0; i < input.length; i++) input[i] = ((i * 2654435761) % 1000) / 1000 - 0.3;
    const outBytes = COUT * w * h * 4;

    // Create the ORT session FIRST so a live session owns the WebGPU device,
    // THEN capture the device and allocate buffers on it. (Allocating on a
    // device from a since-released session gets "cannot be used with Device".)
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
      preferredOutputLocation: 'gpu-buffer', freeDimensionOverrides: { height: h, width: w },
    });
    device = ort.env.webgpu.device;
    hasF16 = device.features.has('shader-f16');
    results.meta.hasF16 = hasF16;
    if (!errHooked) {
      device.addEventListener?.('uncapturederror', (e) => log(`  [GPU uncapturederror] ${e.error?.message || e}`));
      log(`device shader-f16: ${hasF16}`);
      errHooked = true;
    }
    mkBuf = (arr, usage) => {
      const b = device.createBuffer({ size: arr.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, arr);
      return b;
    };

    // shared input + f32 output buffers (ORT baseline + our kernels)
    const inBuf = mkBuf(input, IO);
    const outBuf = device.createBuffer({ size: outBytes, usage: IO });
    const wBuf = mkBuf(weights, GPUBufferUsage.STORAGE);
    const bBuf = mkBuf(bias, GPUBufferUsage.STORAGE);
    const pBuf = mkBuf(new Uint32Array([w, h, CIN, COUT]), GPUBufferUsage.UNIFORM);
    const rbBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // --- ORT fair baseline (gpu-buffer IO) ---
    const feeds = { input: ort.Tensor.fromGpuBuffer(inBuf, { dataType: 'float32', dims: [1, CIN, h, w] }) };
    const fetches = { output: ort.Tensor.fromGpuBuffer(outBuf, { dataType: 'float32', dims: [1, COUT, h, w] }) };

    await session.run(feeds, fetches);
    const ref = new Float32Array(await readBuf(device, outBuf, rbBuf, outBytes));
    const refRelu6 = Float32Array.from(ref, (v) => Math.min(Math.max(v, 0), 6));

    for (let i = 0; i < WARM; i++) await session.run(feeds, fetches);
    await device.queue.onSubmittedWorkDone();
    let t0 = performance.now();
    for (let i = 0; i < ITERS; i++) await session.run(feeds, fetches);
    await device.queue.onSubmittedWorkDone();
    const ortMs = (performance.now() - t0) / ITERS;
    sizeRes.ort.gpuIo = ortMs;
    log(`ORT-webgpu gpu-buffer IO (BASELINE): ${ortMs.toFixed(2)} ms`);
    // NB: keep `session` alive through the kernel runs below — ORT tears the
    // WebGPU device down when its last session is released, which would kill
    // our buffers/pipelines. Release only at end of the size loop.

    // --- our-kernel buffers ---
    const f32Set = [inBuf, wBuf, bBuf, outBuf, pBuf];
    let f16Set = null, outBufH = null;
    if (hasF16) {
      const inH = mkBuf(f32ToF16Bits(input), GPUBufferUsage.STORAGE);
      const wH = mkBuf(f32ToF16Bits(weights), GPUBufferUsage.STORAGE);
      const bH = mkBuf(f32ToF16Bits(bias), GPUBufferUsage.STORAGE);
      outBufH = device.createBuffer({ size: outBytes / 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      f16Set = [inH, wH, bH, outBufH, pBuf];
    } else {
      log('shader-f16 unavailable -> f16 variants skipped');
    }
    const p = { w, h, cin: CIN, cout: COUT };

    for (const k of kernels) {
      const kres = { name: k.name, note: k.note };
      sizeRes.kernels.push(kres);
      if (k.name==='naive' && w>1024) { kres.status='skipped'; log('  naive: skipped (slow at '+w+'p)'); continue; }
      if (k.f16 && !hasF16) { kres.status = 'skipped'; log(`  ${k.name}: skipped (no shader-f16)`); continue; }
      const bufs = k.f16 ? f16Set : f32Set;
      const kOut = k.f16 ? outBufH : outBuf;
      const kOutBytes = k.f16 ? outBytes / 2 : outBytes;
      const expected = k.relu6 ? refRelu6 : ref;
      const tol = k.tol ?? 1e-3;
      try {
        const pipe = device.createComputePipeline({
          layout: 'auto',
          compute: { module: device.createShaderModule({ code: k.code(p) }), entryPoint: 'main' },
        });
        const bind = device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })),
        });
        const run = () => {
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipe);
          pass.setBindGroup(0, bind);
          pass.dispatchWorkgroups(...k.dispatch(p));
          pass.end();
          device.queue.submit([enc.finish()]);
        };
        // correctness vs ORT (clear first so stale output can't mask gaps).
        // Skip the readback at large sizes — each is a 0.5GB map that dominates
        // wall-clock; the kernel is identical code, already gated at 512².
        let maxd = 0;
        if (w <= 1024) {
          const enc = device.createCommandEncoder(); enc.clearBuffer(kOut); device.queue.submit([enc.finish()]);
          run();
          const raw = await readBuf(device, kOut, rbBuf, kOutBytes);
          const got = k.f16 ? f16BitsToF32(new Uint16Array(raw)) : new Float32Array(raw);
          for (let i = 0; i < got.length; i++) { const d = Math.abs(got[i] - expected[i]); if (d > maxd) maxd = d; }
          kres.maxdiff = maxd;
          if (!(maxd <= tol)) { kres.status = 'WRONG'; log(`  ${k.name}: WRONG (maxdiff ${maxd.toExponential(2)}, tol ${tol})`); continue; }
        } else {
          kres.maxdiff = 'skipped@large';
        }
        for (let i = 0; i < WARM; i++) run();
        await device.queue.onSubmittedWorkDone();
        t0 = performance.now();
        for (let i = 0; i < ITERS; i++) run();
        await device.queue.onSubmittedWorkDone();
        const ms = (performance.now() - t0) / ITERS;
        kres.status = 'ok'; kres.ms = ms; kres.speedup = ortMs / ms;
        log(`  ${k.name}: ${ms.toFixed(2)} ms (${(ortMs / ms).toFixed(2)}x vs ORT, maxdiff ${maxd.toExponential(1)})`);
      } catch (e) {
        kres.status = 'FAILED'; kres.error = e.message;
        log(`  ${k.name}: FAILED — ${e.message}`);
      }
    }
    const extra = f16Set ? [f16Set[0], f16Set[1], f16Set[2], outBufH] : [];
    [inBuf, outBuf, wBuf, bBuf, pBuf, rbBuf, ...extra].forEach((b) => b.destroy());
    // release only now that this size's kernels are done (keeps device alive)
    session.release?.();
    log('');
  }
  log('DONE');
  window.__done = true;
}
main().catch((e) => { log('ERROR: ' + (e.stack || e.message)); results.error = String(e.stack || e.message); window.__done = true; });
