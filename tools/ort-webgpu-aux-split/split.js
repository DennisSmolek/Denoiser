// Does the "compute enc_conv0 ourselves, let ORT-WebGPU run the rest" workaround
// actually CLEAR the aux speckle? This tests the premise before we build the
// real WGSL kernel — because the premise is an assumption, not a fact.
//
// Companion to ../ort-webgpu-aux-repro (which proved the bug is ORT-web's WebGPU
// EP). Here we SPLIT the 9ch aux U-Net at enc_conv0:
//   head = enc_conv0 (Conv+relu6): input(9ch) -> enc_conv0_relu6_2 (32ch)
//   tail = enc_conv1 .. output    (topology-identical to the 3ch model's tail)
//
// IMPORTANT graph fact: the OIDN U-Net feeds the RAW 9ch input into TWO convs —
// enc_conv0 (first layer) AND dec_conv1a (via the concat_38 input-skip near the
// output). So the tail STILL ingests the raw 9ch input at dec_conv1a. If the ORT
// WebGPU bug is "any conv reducing the raw >3ch input", externalising only
// enc_conv0 is NOT enough and the tail will still diverge. This test decides it.
//
// WASM (CPU) is the correctness oracle (matches native OIDN). enc_conv0 run on
// WASM stands in for our WGSL kernel (which is correct to 2e-6).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

const el = document.getElementById('log');
const log = (m) => { el.textContent += m + '\n'; console.log(m); };
window.__results = [];
window.__done = false;

const C = 9, H = 256, W = 256, C1 = 32; // aux input, enc_conv0 out channels

// Same deterministic input as the sibling repro, so results are comparable.
function makeInput(C, H, W) {
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

function localVar(arr, H, W) {
  let acc = 0, n = 0;
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      let s = 0, s2 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const v = arr[(y + dy) * W + (x + dx)]; s += v; s2 += v * v;
      }
      const m = s / 9; acc += s2 / 9 - m * m; n++;
    }
  return acc / n;
}

const bytesCache = {};
async function bytes(file) {
  if (!bytesCache[file]) bytesCache[file] = new Uint8Array(await (await fetch(`./models/${file}`)).arrayBuffer());
  return bytesCache[file];
}

// Run one session. feeds: { name: Float32Array-or-Tensor }. Returns first output
// as Float32Array plus timing.
async function run(file, ep, feeds) {
  const session = await ort.InferenceSession.create(await bytes(file), {
    executionProviders: [ep], graphOptimizationLevel: 'all',
  });
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  const t = out[session.outputNames[0]];
  const data = Float32Array.from(t.data);
  await session.release?.();
  return { data, ms };
}

// Compare a candidate output against the WASM reference (channel-0 plane metrics).
function compare(label, cand, ref) {
  const a = cand, b = ref, plane = H * W;
  let maxd = 0, sum = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); sum += d; if (d > maxd) maxd = d; }
  const lvC = localVar(a.subarray(0, plane), H, W);
  const lvR = localVar(b.subarray(0, plane), H, W);
  const res = {
    label, maxAbsDiff: maxd, meanAbsDiff: sum / a.length,
    candMin: Math.min(...a.subarray(0, plane)), refMin: Math.min(...b.subarray(0, plane)),
    localVarCand: lvC, localVarRef: lvR, noiseRatio: lvC / lvR,
  };
  window.__results.push(res);
  const verdict = maxd > 1e-2 ? '❌ DIVERGES' : '✅ matches reference';
  log(`\n${label}`);
  log(`  max|Δ| vs WASM-ref: ${maxd.toExponential(3)}   mean|Δ|: ${res.meanAbsDiff.toExponential(3)}`);
  log(`  noise (local-var) ratio vs ref: ${res.noiseRatio.toFixed(2)}x   (cand min ${res.candMin.toFixed(4)} vs ref ${res.refMin.toFixed(4)})`);
  log(`  ${verdict}`);
  return res;
}

async function main() {
  log('onnxruntime-web ' + ort.env.versions.web + ' | split-graph workaround test (base 9ch aux)');
  log(`input ${C}ch ${W}x${H}, enc_conv0 out ${C1}ch. WASM = correctness oracle.\n`);
  const input = makeInput(C, H, W);
  const inputT = () => new ort.Tensor('float32', input, [1, C, H, W]);

  // 0) reference: full model on WASM (matches native OIDN)
  const ref = (await run('full.onnx', 'wasm', { input: inputT() })).data;
  log('reference = full model on WASM  ✓ computed');

  // 1) known-bad baseline: full model on WebGPU (should diverge ~1e-1)
  const fullGpu = (await run('full.onnx', 'webgpu', { input: inputT() })).data;
  compare('[baseline] full model on WebGPU (the bug)', fullGpu, ref);

  // 2) our correct enc_conv0 (run head on WASM = stand-in for our WGSL kernel)
  const F = (await run('head_enc0.onnx', 'wasm', { input: inputT() })).data;
  const Ftensor = () => new ort.Tensor('float32', F, [1, C1, H, W]);
  log(`\nenc_conv0 computed on WASM (our kernel's correct output), ${F.length} vals`);

  // 3) sanity: tail on WASM with correct inputs must reproduce the reference
  const tailWasm = (await run('tail.onnx', 'wasm', { enc_conv0_relu6_2: Ftensor(), input: inputT() })).data;
  compare('[sanity] split faithful? tail on WASM (should match)', tailWasm, ref);

  // 4) THE TEST: enc_conv0 done correctly off-GPU, everything else on ORT-WebGPU.
  //    Tail still ingests the raw 9ch input at dec_conv1a (concat_38 skip).
  const tailGpu = (await run('tail.onnx', 'webgpu', { enc_conv0_relu6_2: Ftensor(), input: inputT() })).data;
  compare('[WORKAROUND] enc_conv0 off-GPU + tail on WebGPU', tailGpu, ref);

  log('\nDONE');
  window.__done = true;
}

main().catch((e) => { log('ERROR: ' + (e.stack || e.message)); window.__done = true; });
