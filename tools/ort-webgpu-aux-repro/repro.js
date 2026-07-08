// Minimal, self-contained repro: does onnxruntime-web's WebGPU EP miscompute the
// OIDN 9-channel aux U-Net vs its own WASM (CPU) EP, on the SAME fixed input?
//
// Deliberately uses BARE vanilla sessions — no gpu-buffer IO, no
// freeDimensionOverrides, no denoiser-engine code, plain CPU-array in/out — so
// the ONLY variable is the execution provider. If WebGPU != WASM here, the
// denoiser library is not involved and the defect is in ORT-web's WebGPU EP.
//
// Everything is fetched from CDNs (ORT + models), so this file is portable —
// attach it to an onnxruntime issue as-is.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs';

const MODELS_CDN = 'https://cdn.jsdelivr.net/gh/pmndrs/denoiser-weights@models-v1/models';
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

const el = document.getElementById('log');
const log = (m) => { el.textContent += m + '\n'; console.log(m); };
window.__results = [];
window.__done = false;

// Deterministic, structured input in [0,1] (a smooth gradient + a little
// per-channel noise) — realistic enough for a denoiser conv net, and identical
// on every machine so results are comparable. NCHW [1, C, H, W].
function makeInput(C, H, W) {
  const a = new Float32Array(C * H * W);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const g = (x / W) * 0.6 + (y / H) * 0.3;                 // smooth base
        const n = (((x * 73856093) ^ (y * 19349663) ^ (c * 83492791)) % 1000) / 1000; // hash noise
        a[c * H * W + y * W + x] = Math.min(1, Math.max(0, g + (n - 0.5) * 0.25));
      }
    }
  }
  return a;
}

function localVar(arr, C, H, W) {
  // mean 3x3 local variance over channel 0 — the "noise" metric
  let acc = 0, n = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      let s = 0, s2 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const v = arr[(y + dy) * W + (x + dx)]; s += v; s2 += v * v;
      }
      const m = s / 9; acc += s2 / 9 - m * m; n++;
    }
  }
  return acc / n;
}

// f32 -> IEEE half bits (Uint16), for fp16-IO models.
function f32ToF16Bits(src) {
  const out = new Uint16Array(src.length);
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i]; const x = u[0], sign = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 112, m = (x >>> 13) & 0x3ff;
    out[i] = e <= 0 ? sign : e >= 31 ? (sign | 0x7c00) : (sign | (e << 10) | m);
  }
  return out;
}
function f16BitsToF32(bits) {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i], s = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

async function runEP(modelBytes, ep, input, C, H, W, fp16) {
  const opts = { executionProviders: [ep], graphOptimizationLevel: 'all' };
  const session = await ort.InferenceSession.create(modelBytes, opts);
  const tensor = fp16
    ? new ort.Tensor('float16', f32ToF16Bits(input), [1, C, H, W])
    : new ort.Tensor('float32', input, [1, C, H, W]);
  const feeds = { input: tensor };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  const t = out[session.outputNames[0]];
  // fp16 output arrives either as Uint16Array (half bits) or Float16Array (real
  // values) depending on ORT-web version; only the former needs bit-decoding.
  const data = (t.data instanceof Uint16Array) ? f16BitsToF32(t.data) : Float32Array.from(t.data);
  await session.release?.();
  return { data, ms };
}

async function testModel(label, file, C, H, W) {
  const fp16 = file.includes('.fp16.');
  log(`\n=== ${label}  (${C}ch, ${W}x${H}${fp16 ? ', fp16' : ''}) ===`);
  const modelBytes = new Uint8Array(await (await fetch(`${MODELS_CDN}/${file}`)).arrayBuffer());
  const input = makeInput(C, H, W);

  let wasm, webgpu;
  try { wasm = await runEP(modelBytes, 'wasm', input, C, H, W, fp16); }
  catch (e) { log(`  WASM  FAILED: ${e.message}`); return; }
  try { webgpu = await runEP(modelBytes, 'webgpu', input, C, H, W, fp16); }
  catch (e) { log(`  WebGPU FAILED: ${e.message}`); return; }

  const a = webgpu.data, b = wasm.data; // WASM (CPU) is the reference
  let maxd = 0, sum = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); sum += d; if (d > maxd) maxd = d; }
  const outHW = a.length / C; // COUT*H*W but COUT=3 for these models -> use H*W plane below
  const plane = H * W;
  const lvGpu = localVar(a.subarray(0, plane), 1, H, W);
  const lvWasm = localVar(b.subarray(0, plane), 1, H, W);

  const res = {
    label, file, C, W, H,
    maxAbsDiff: maxd, meanAbsDiff: sum / a.length,
    webgpuMin: Math.min(...a.subarray(0, plane)),
    wasmMin: Math.min(...b.subarray(0, plane)),
    localVarWebGPU: lvGpu, localVarWASM: lvWasm,
    noiseRatio: lvGpu / lvWasm,
    wasmMs: wasm.ms, webgpuMs: webgpu.ms,
  };
  window.__results.push(res);
  log(`  max|Δ| WebGPU-vs-WASM: ${maxd.toExponential(3)}   mean|Δ|: ${res.meanAbsDiff.toExponential(3)}`);
  log(`  output local-variance: WebGPU ${lvGpu.toExponential(3)} vs WASM ${lvWasm.toExponential(3)}  (ratio ${res.noiseRatio.toFixed(1)}x)`);
  log(`  output min: WebGPU ${res.webgpuMin.toFixed(4)}  WASM ${res.wasmMin.toFixed(4)}`);
  const verdict = maxd > 1e-2 ? '❌ DIVERGES (WebGPU EP miscomputes this model)' : '✅ matches';
  log(`  ${verdict}`);
}

async function main() {
  log('onnxruntime-web ' + ort.env.versions.web + ' | vanilla sessions, CPU-array IO, EP is the only variable');
  // control: 3-channel color model (known-good in our stack)
  await testModel('color / rt_hdr (CONTROL, 3ch)', 'rt_hdr.onnx', 3, 256, 256);
  // threshold probe: 6-channel (color+albedo) — is it >3ch generally, or only 9?
  await testModel('aux / rt_hdr_alb (6ch)', 'rt_hdr_alb.onnx', 6, 256, 256);
  // suspects: the 9-channel aux nets
  await testModel('aux / rt_hdr_calb_cnrm (base, 9ch)', 'rt_hdr_calb_cnrm.onnx', 9, 256, 256);
  await testModel('aux / rt_hdr_calb_cnrm_small (9ch)', 'rt_hdr_calb_cnrm_small.onnx', 9, 256, 256);
  // fp16 variant — different shader path; does it dodge the bug?
  await testModel('aux / rt_hdr_calb_cnrm.fp16 (9ch, fp16)', 'rt_hdr_calb_cnrm.fp16.onnx', 9, 256, 256);
  log('\nDONE');
  window.__done = true;
}

main().catch((e) => { log('ERROR: ' + (e.stack || e.message)); window.__done = true; });
