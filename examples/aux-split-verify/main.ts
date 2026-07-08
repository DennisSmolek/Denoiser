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

async function main() {
  log('onnxruntime-web ' + ort.env.versions.web + ' | real DenoiseEngine, 9ch aux, 256x256');
  const vals = makeVals();
  const color = toRGBA(vals, 0), albedo = toRGBA(vals, 3), normal = toRGBA(vals, 6);

  const [fullBytes, tailBytes, encBuf] = await Promise.all([
    fetch('models/full.onnx').then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
    fetch('models/tail.onnx').then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b)),
    fetch('models/enc0.bin').then((r) => r.arrayBuffer()),
  ]);
  const encF = new Float32Array(encBuf);
  const encWeights = encF.slice(0, C1 * C * 9);
  const encBias = encF.slice(C1 * C * 9, C1 * C * 9 + C1);
  log(`enc0 weights ${encWeights.length}, bias ${encBias.length}`);

  // WASM full-model reference on the identical bytes/255 input
  const ref = refToBytes(await wasmFullReference(fullBytes, nchwFrom(color, albedo, normal)));
  log('WASM full-model reference computed');

  const denoiseOpts = { albedo, normal, srgb: false, hdr: false } as const;

  // (B) baseline: full model on ORT-WebGPU through the real engine (the bug)
  const fullEngine = await DenoiseEngine.create(fullBytes, { channels: 9 });
  const outFull = await fullEngine.denoise(color, W, H, denoiseOpts);

  // (A) the fix: split mode through the real engine
  const splitEngine = await DenoiseEngine.create(tailBytes, {
    channels: 9,
    split: { encWeights, encBias, encOutChannels: C1 },
  });
  const outSplit = await splitEngine.denoise(color, W, H, denoiseOpts);

  const dFull = diff(outFull, ref);
  const dSplit = diff(outSplit, ref);
  const nRef = noise(ref), nFull = noise(outFull), nSplit = noise(outSplit);

  log('\n[BASELINE] full model on ORT-WebGPU (the bug) vs WASM reference:');
  log(`  max byte Δ ${dFull.maxByteDiff}  mean ${dFull.meanByteDiff.toFixed(3)}  pixels >2 LSB: ${dFull.pctOver2LSB.toFixed(2)}%`);
  log('\n[FIX] split engine (WGSL enc_conv0 + tail on WebGPU) vs WASM reference:');
  log(`  max byte Δ ${dSplit.maxByteDiff}  mean ${dSplit.meanByteDiff.toFixed(3)}  pixels >2 LSB: ${dSplit.pctOver2LSB.toFixed(2)}%`);
  log(`\nnoise (R local-var): reference ${nRef.toFixed(2)}  full-webgpu ${nFull.toFixed(2)}  split ${nSplit.toFixed(2)}`);

  const pass = dSplit.maxByteDiff <= 3 && dFull.maxByteDiff > dSplit.maxByteDiff + 3;
  log(`\nVERDICT: ${pass ? '✅ split matches reference AND fixes the baseline divergence' : '❌ unexpected'}`);

  (window as any).__results = { dFull, dSplit, noise: { nRef, nFull, nSplit }, pass };
  show('WASM reference', ref);
  show('full WebGPU (bug)', outFull);
  show('split (fix)', outSplit);

  splitEngine.destroy(); // releasing last session tears down the shared device
  log('\nDONE');
  (window as any).__done = true;
}

main().catch((e) => { log('ERROR: ' + (e.stack || e.message)); (window as any).__done = true; });
