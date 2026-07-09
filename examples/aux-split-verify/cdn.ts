// Shipped-default integration test: the high-level Denoiser with NO config —
// default weightsUrl (jsDelivr @models-v2), splitAux default on — must fetch the
// split artifacts from the live CDN and denoise 9ch aux cleanly. Compared to an
// explicit splitAux:false Denoiser (the plain, speckled path). No local models.
import { Denoiser } from '../../packages/denoiser/src/index';

const W = 256, H = 256;
const el = document.getElementById('log')!;
const log = (m: string) => { el.textContent += m + '\n'; console.log(m); };
(window as any).__done = false;

// capture the fallback warning ("splitAux artifacts unavailable") to prove the
// split path was actually taken (not silently falling back to the plain model).
const warnings: string[] = [];
const origWarn = console.warn.bind(console);
console.warn = (...a: unknown[]) => { warnings.push(a.map(String).join(' ')); origWarn(...a); };

function img(fill: (x: number, y: number, c: number) => number): ImageData {
  const d = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      for (let c = 0; c < 3; c++) d[i + c] = Math.round(fill(x, y, c) * 255);
      d[i + 3] = 255;
    }
  return new ImageData(d, W, H);
}
const hash = (x: number, y: number, c: number) => (((x * 73856093) ^ (y * 19349663) ^ (c * 83492791)) % 1000) / 1000;
const color = img((x, y, c) => Math.min(1, Math.max(0, (x / W) * 0.6 + (y / H) * 0.3 + (hash(x, y, c) - 0.5) * 0.35))); // noisy
const albedo = img((x, y) => (x / W) * 0.6 + (y / H) * 0.3);       // clean
const normal = img((x, y, c) => c === 2 ? 1 : 0.5 + 0.2 * Math.sin((x + y) / 24)); // encoded-ish

function noise(d: Uint8ClampedArray): number {
  let acc = 0, n = 0;
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      let s = 0, s2 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const v = d[((y + dy) * W + (x + dx)) * 4]; s += v; s2 += v * v;
      }
      const m = s / 9; acc += s2 / 9 - m * m; n++;
    }
  return acc / n;
}

async function main() {
  log('High-level Denoiser, default weightsUrl (jsDelivr @models-v2), 9ch aux.\n');

  // (1) shipped default — splitAux on, fetches tail/enc0 from the CDN
  const d1 = await Denoiser.create({});
  log(`default Denoiser ready. model resolves to a cleanAux net; splitAux default on.`);
  const wBefore = warnings.length;
  const out1 = await d1.denoise(color, { albedo, normal });
  const usedFallback = warnings.slice(wBefore).some((w) => /splitAux artifacts unavailable/.test(w));

  // (2) explicit splitAux:false — plain model on WebGPU (the bug)
  const d2 = await Denoiser.create({ splitAux: false });
  const out2 = await d2.denoise(color, { albedo, normal });

  const nSplit = out1 ? noise(out1.data) : NaN;
  const nPlain = out2 ? noise(out2.data) : NaN;
  log(`\nsplit path fetched from CDN (no fallback): ${!usedFallback ? 'YES ✅' : 'NO ❌ (fell back)'}`);
  log(`output noise (R local-var):  split ${nSplit.toFixed(1)}   plain/bug ${nPlain.toFixed(1)}`);
  const pass = !usedFallback && nSplit < nPlain * 0.85;
  log(`\nVERDICT: ${pass ? '✅ shipped default fetches artifacts from CDN and denoises aux cleaner than the plain path' : '❌ unexpected'}`);

  (window as any).__results = { usedFallback, nSplit, nPlain, pass, warnings };
  d2.destroyDevice(); // releases the shared ORT device (both denoisers share it)
  log('\nDONE');
  (window as any).__done = true;
}

main().catch((e) => { log('ERROR: ' + (e.stack || e.message)); (window as any).__done = true; });
