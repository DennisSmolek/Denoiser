// Real-engine verification of the aux split-graph workaround.
//
// We drive the ACTUAL DenoiseEngine (not a standalone kernel) through its CPU
// denoise() path with srgb:false, hdr:false, so the NCHW the engine feeds the
// model is exactly bytes/255 for all 9 channels — deterministic and
// reproducible. That lets us build a WASM full-model reference on the identical
// input and check that:
//   (A) split engine (our WGSL enc_conv0 + tail on ORT-WebGPU) == WASM reference
//       -> the workaround is correctly wired AND clears the bug, and
//   (B) normal engine (full model on ORT-WebGPU) != reference (the speckle bug),
// proving the fix on the real runtime.
import * as ort from 'onnxruntime-web/webgpu';
import { DenoiseEngine } from '../../packages/denoiser/src/ort/engine';

const W = 256, H = 256, C = 9, C1 = 32;
const el = document.getElementById('log')!;
const views = document.getElementById('views')!;
const log = (m: string) => { el.textContent += m + '\n'; console.log(m); };
(window as any).__results = {};
(window as any).__done = false;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

// Same deterministic field as the ORT repro tests (gradient + hash noise), in
// [0,1]. Channels 0-2 color (noisy), 3-5 albedo, 6-8 normal.
function makeVals(): Float32Array {
  const a = new Float32Array(C * H * W);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const g = (x / W) * 0.6 + (y / H) * 0.3;
        const n = (((x * 73856093) ^ (y * 19349663) ^ (c * 83492791)) % 1000) / 1000;
        a[c * H * W + y * W + x] = Math.min(1, Math.max(0, g + (n - 0.5) * 0.25));
      }
  return a;
}

// Pack a 3-channel group into RGBA8 (what the engine's denoise() ingests).
function toRGBA(vals: Float32Array, c0: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      out[i * 4 + 0] = Math.round(vals[(c0 + 0) * H * W + i] * 255);
      out[i * 4 + 1] = Math.round(vals[(c0 + 1) * H * W + i] * 255);
      out[i * 4 + 2] = Math.round(vals[(c0 + 2) * H * W + i] * 255);
      out[i * 4 + 3] = 255;
    }
  return out;
}

// Rebuild the engine's NCHW input (bytes/255) from the same RGBA8 groups, so the
// WASM reference sees byte-identical input.
function nchwFrom(color: Uint8ClampedArray, albedo: Uint8ClampedArray, normal: Uint8ClampedArray): Float32Array {
  const nchw = new Float32Array(C * H * W);
  const grp = [color, albedo, normal];
  for (let g = 0; g < 3; g++)
    for (let ch = 0; ch < 3; ch++)
      for (let i = 0; i < W * H; i++)
        nchw[(g * 3 + ch) * H * W + i] = grp[g][i * 4 + ch] / 255;
  return nchw;
}

async function wasmFullReference(fullBytes: Uint8Array, nchw: Float32Array): Promise<Float32Array> {
  const s = await ort.InferenceSession.create(fullBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  const out = await s.run({ input: new ort.Tensor('float32', nchw, [1, C, H, W]) });
  const t = out[s.outputNames[0]];
  await s.release?.();
  return Float32Array.from(t.data as Float32Array);
}

// Engine returns RGBA8 (already clamp01*255). Reference model output is 3ch
// float -> clamp01*255 the same way. Compare RGB bytes.
function refToBytes(refCHW: Float32Array): Uint8ClampedArray {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    for (let ch = 0; ch < 3; ch++) out[i * 4 + ch] = Math.round(Math.min(1, Math.max(0, refCHW[ch * H * W + i])) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

function diff(cand: Uint8ClampedArray, ref: Uint8ClampedArray) {
  let maxd = 0, sum = 0, n = 0, over2 = 0;
  for (let i = 0; i < W * H; i++)
    for (let ch = 0; ch < 3; ch++) {
      const d = Math.abs(cand[i * 4 + ch] - ref[i * 4 + ch]);
      sum += d; n++; if (d > maxd) maxd = d; if (d > 2) over2++;
    }
  return { maxByteDiff: maxd, meanByteDiff: sum / n, pctOver2LSB: (100 * over2) / n };
}

// local variance of the R channel — a noise proxy (speckle -> high)
function noise(bytes: Uint8ClampedArray) {
  let acc = 0, n = 0;
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      let s = 0, s2 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const v = bytes[((y + dy) * W + (x + dx)) * 4]; s += v; s2 += v * v;
      }
      const m = s / 9; acc += s2 / 9 - m * m; n++;
    }
  return acc / n;
}

function show(title: string, bytes: Uint8ClampedArray) {
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  cv.getContext('2d')!.putImageData(new ImageData(bytes, W, H), 0, 0);
  const fig = document.createElement('figure');
  fig.appendChild(cv);
  const cap = document.createElement('figcaption');
  cap.textContent = title;
  fig.appendChild(cap);
  views.appendChild(fig);
}

async function fetchBytes(url: string) { return new Uint8Array(await (await fetch(url)).arrayBuffer()); }

async function runPrecision(
  precision: 'fp32' | 'fp16', ref: Uint8ClampedArray,
  color: Uint8ClampedArray, albedo: Uint8ClampedArray, normal: Uint8ClampedArray,
  keep: DenoiseEngine[],
) {
  const sfx = precision === 'fp16' ? '.fp16' : '';
  const [fullBytes, tailBytes, encBuf] = await Promise.all([
    fetchBytes(`models/full${sfx}.onnx`),
    fetchBytes(`models/tail${sfx}.onnx`),
    fetch(`models/enc0${sfx}.bin`).then((r) => r.arrayBuffer()),
  ]);
  const encF = new Float32Array(encBuf);
  const encWeights = encF.slice(0, C1 * C * 9);
  const encBias = encF.slice(C1 * C * 9, C1 * C * 9 + C1);
  const denoiseOpts = { albedo, normal, srgb: false, hdr: false } as const;

  let outFull: Uint8ClampedArray, outSplit: Uint8ClampedArray;
  try {
    const fullEngine = await DenoiseEngine.create(fullBytes, { channels: 9, precision });
    keep.push(fullEngine);
    outFull = await fullEngine.denoise(color, W, H, denoiseOpts);
    const splitEngine = await DenoiseEngine.create(tailBytes, {
      channels: 9, precision, split: { encWeights, encBias, encOutChannels: C1 },
    });
    keep.push(splitEngine);
    outSplit = await splitEngine.denoise(color, W, H, denoiseOpts);
  } catch (e) {
    log(`\n[${precision}] SKIPPED: ${(e as Error).message}`);
    return null;
  }

  const dFull = diff(outFull, ref), dSplit = diff(outSplit, ref);
  const nRef = noise(ref), nFull = noise(outFull), nSplit = noise(outSplit);
  log(`\n=== ${precision} (vs fp32 WASM/native reference) ===`);
  log(`  [baseline] full model on WebGPU (bug):  max Δ ${dFull.maxByteDiff}  mean ${dFull.meanByteDiff.toFixed(3)}  >2LSB ${dFull.pctOver2LSB.toFixed(1)}%  noise ${nFull.toFixed(1)}`);
  log(`  [FIX] split (WGSL enc_conv0 + tail):     max Δ ${dSplit.maxByteDiff}  mean ${dSplit.meanByteDiff.toFixed(3)}  >2LSB ${dSplit.pctOver2LSB.toFixed(1)}%  noise ${nSplit.toFixed(1)}`);
  const tol = precision === 'fp16' ? 12 : 3; // fp16 adds precision noise vs the fp32 reference
  const pass = dSplit.maxByteDiff <= tol && dFull.maxByteDiff > dSplit.maxByteDiff + 3;
  log(`  VERDICT: ${pass ? '✅ split clean, baseline speckled' : '❌ unexpected'}`);
  show(`${precision} ref`, ref);
  show(`${precision} full (bug)`, outFull);
  show(`${precision} split (fix)`, outSplit);
  return { precision, dFull, dSplit, noise: { nRef, nFull, nSplit }, pass };
}

async function main() {
  log('onnxruntime-web ' + ort.env.versions.web + ' | real DenoiseEngine, 9ch aux, 256x256');
  const vals = makeVals();
  const color = toRGBA(vals, 0), albedo = toRGBA(vals, 3), normal = toRGBA(vals, 6);

  // fp32 WASM full-model reference on the identical bytes/255 input = native truth
  const fullBytes = await fetchBytes('models/full.onnx');
  const ref = refToBytes(await wasmFullReference(fullBytes, nchwFrom(color, albedo, normal)));
  log('fp32 WASM full-model reference computed (native truth)');

  const keep: DenoiseEngine[] = [];
  const results = [
    await runPrecision('fp32', ref, color, albedo, normal, keep),
    await runPrecision('fp16', ref, color, albedo, normal, keep),
  ].filter(Boolean);

  (window as any).__results = results;
  keep.forEach((e) => e.destroy());
  log('\nDONE');
  (window as any).__done = true;
}

main().catch((e) => { log('ERROR: ' + (e.stack || e.message)); (window as any).__done = true; });
