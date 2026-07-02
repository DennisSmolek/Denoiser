// Benchmark harness for the denoiser package (Phase 0 of perf-plan.md).
// Deterministic seeded noise images, cold + warm timings, per-stage stats from
// Denoiser.lastStats, PSNR parity between precisions. Results land in #results
// (table) and #json (machine-readable, window.__benchResults).
import { Denoiser } from 'denoiser';

const params = new URLSearchParams(location.search);
const only = params.get('only');
const batchParam = params.get('batch') ? Number(params.get('batch')) : undefined;
const SCENARIOS = [
  { label: '512x512', w: 512, h: 512 },
  { label: '1280x720', w: 1280, h: 720 },
  { label: '1920x1080', w: 1920, h: 1080 },
].filter((s) => !only || s.label === only);
const WARM_RUNS = 5;

const status = document.querySelector<HTMLPreElement>('#status')!;
const jsonOut = document.querySelector<HTMLPreElement>('#json')!;
const tbody = document.querySelector<HTMLTableSectionElement>('#results tbody')!;
const noisyCanvas = document.querySelector<HTMLCanvasElement>('#noisy')!;
const cleanCanvas = document.querySelector<HTMLCanvasElement>('#clean')!;
const precisionSel = document.querySelector<HTMLSelectElement>('#precision')!;
const qualitySel = document.querySelector<HTMLSelectElement>('#quality')!;
const runAllBtn = document.querySelector<HTMLButtonElement>('#runAll')!;

const log = (m: string) => { status.textContent += m + '\n'; console.log(m); };

// Deterministic PRNG so every run/phase sees identical input pixels.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const noisyCache = new Map<string, ImageData>();
function makeNoisy(w: number, h: number): ImageData {
  const key = `${w}x${h}`;
  const cached = noisyCache.get(key);
  if (cached) return cached;
  const rand = mulberry32(1234567);
  const img = new ImageData(w, h);
  const d = img.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      // base scene: two-way gradient + a few disks, so the denoiser has structure to keep
      let r = 40 + (215 * x) / w;
      let g = 60 + (160 * y) / h;
      let b = 200 - (140 * x) / w;
      const disks = [
        [w * 0.3, h * 0.35, Math.min(w, h) * 0.18, 229, 62, 62],
        [w * 0.68, h * 0.6, Math.min(w, h) * 0.22, 56, 161, 105],
        [w * 0.55, h * 0.25, Math.min(w, h) * 0.1, 236, 201, 75],
      ];
      for (const [cx, cy, rad, dr, dg, db] of disks) {
        if ((x - cx) ** 2 + (y - cy) ** 2 < rad * rad) { r = dr; g = dg; b = db; }
      }
      // approx-gaussian noise (sum of 3 uniforms), sigma ~40 — matches the old test example
      const n = () => (rand() + rand() + rand() - 1.5) * 40;
      d[i] = Math.max(0, Math.min(255, r + n()));
      d[i + 1] = Math.max(0, Math.min(255, g + n()));
      d[i + 2] = Math.max(0, Math.min(255, b + n()));
      d[i + 3] = 255;
    }
  }
  noisyCache.set(key, img);
  return img;
}

function psnr(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let se = 0;
  let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; se += d * d; n++; }
  }
  const mse = se / n;
  return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
}

const median = (xs: number[]) => [...xs].sort((p, q) => p - q)[Math.floor(xs.length / 2)];

interface Result {
  scenario: string; precision: string; quality: string; tiles: number;
  buildMs: number; coldMs: number; warmMs: number;
  uploadMs: number; encodeMs: number; runMs: number; resolveMs: number;
  psnrVsFp32: number | null;
}
const results: Result[] = [];
(window as unknown as { __benchResults: Result[] }).__benchResults = results;
// fp32 outputs kept per scenario+quality so a later fp16 pass can compute parity
const fp32Outputs = new Map<string, Uint8ClampedArray>();

function render(res: Result) {
  results.push(res);
  const tr = document.createElement('tr');
  const f = (x: number | null) => (x == null ? '—' : x === Infinity ? 'inf' : x.toFixed(1));
  tr.innerHTML = `<td>${res.scenario} ${res.precision} ${res.quality}</td><td>${res.tiles}</td>` +
    [res.buildMs, res.coldMs, res.warmMs, res.uploadMs, res.encodeMs, res.runMs, res.resolveMs, res.psnrVsFp32]
      .map((x) => `<td>${f(x)}</td>`).join('');
  tbody.appendChild(tr);
  jsonOut.textContent = JSON.stringify(results, null, 1);
}

async function benchScenario(w: number, h: number, label: string): Promise<Result> {
  const precision = precisionSel.value as 'fp32' | 'fp16';
  const quality = qualitySel.value as 'fast' | 'balanced';
  log(`--- ${label} (${precision}, ${quality}) ---`);

  const denoiser = new Denoiser({ precision, batch: batchParam });
  denoiser.weightsUrl = '/models';
  denoiser.quality = quality;

  const noisy = makeNoisy(w, h);
  noisyCanvas.width = w; noisyCanvas.height = h;
  noisyCanvas.getContext('2d')!.putImageData(noisy, 0, 0);

  const tb = performance.now();
  denoiser.setInputData('color', noisy.data, w, h);
  await denoiser.build();
  const buildMs = performance.now() - tb;
  log(`graph capture: ${(denoiser as unknown as { engine?: { graphCaptured?: boolean } })['engine']?.graphCaptured ?? 'n/a'}`);

  const tc = performance.now();
  let out = (await denoiser.execute(noisy)) as ImageData;
  const coldMs = performance.now() - tc;

  const warm: number[] = [];
  const stages = { uploadMs: [] as number[], encodeMs: [] as number[], runMs: [] as number[], resolveMs: [] as number[] };
  for (let i = 0; i < WARM_RUNS; i++) {
    const t0 = performance.now();
    out = (await denoiser.execute(noisy)) as ImageData;
    warm.push(performance.now() - t0);
    const s = denoiser.lastStats!;
    stages.uploadMs.push(s.uploadMs); stages.encodeMs.push(s.encodeMs);
    stages.runMs.push(s.runMs); stages.resolveMs.push(s.resolveMs);
  }

  cleanCanvas.width = w; cleanCanvas.height = h;
  cleanCanvas.getContext('2d')!.putImageData(out, 0, 0);

  const key = `${label}|${quality}`;
  let psnrVsFp32: number | null = null;
  if (precision === 'fp32') fp32Outputs.set(key, out.data);
  else if (fp32Outputs.has(key)) psnrVsFp32 = psnr(fp32Outputs.get(key)!, out.data);

  const tiles = denoiser.tileInfo?.tilesX! * denoiser.tileInfo?.tilesY!;
  const res: Result = {
    scenario: label, precision, quality, tiles,
    buildMs, coldMs, warmMs: median(warm),
    uploadMs: median(stages.uploadMs), encodeMs: median(stages.encodeMs),
    runMs: median(stages.runMs), resolveMs: median(stages.resolveMs),
    psnrVsFp32,
  };
  log(`build ${buildMs.toFixed(0)}ms | cold ${coldMs.toFixed(0)}ms | warm ${res.warmMs.toFixed(1)}ms ` +
    `(${tiles} tiles: run ${res.runMs.toFixed(1)} + encode ${res.encodeMs.toFixed(1)} + resolve ${res.resolveMs.toFixed(1)})`);
  denoiser.dispose();
  render(res);
  return res;
}

async function runAll() {
  runAllBtn.disabled = true;
  try {
    for (const s of SCENARIOS) await benchScenario(s.w, s.h, s.label);
    log('ALL DONE');
  } catch (e) {
    log('ERROR: ' + (e as Error).message);
    console.error(e);
  } finally {
    runAllBtn.disabled = false;
  }
}

// Ad-hoc single run for automated verification (returns pixels + stage stats).
(window as unknown as Record<string, unknown>).__denoiseOnce = async (
  w: number, h: number,
  opts: { precision?: 'fp32' | 'fp16'; batch?: number; quality?: 'fast' | 'balanced' } = {},
) => {
  const dn = new Denoiser({ precision: opts.precision ?? 'fp32', batch: opts.batch });
  dn.weightsUrl = '/models';
  dn.quality = opts.quality ?? 'fast';
  const out = (await dn.execute(makeNoisy(w, h))) as ImageData;
  const stats = dn.lastStats;
  dn.dispose();
  return { data: out.data, stats };
};
(window as unknown as Record<string, unknown>).__psnr = psnr;

async function main() {
  if (!('gpu' in navigator)) { log('ERROR: WebGPU not available.'); return; }
  log(`bench ready — pick precision/quality and hit Run.${batchParam ? ` (batch=${batchParam})` : ''}`);
  runAllBtn.disabled = false;
  runAllBtn.addEventListener('click', runAll);
}

main().catch((e) => log('ERROR: ' + (e as Error).message));
