import { Denoiser } from 'denoiser';

async function run(): Promise<Denoiser> {
  const noisyImg = document.querySelector<HTMLImageElement>('#noisy-img')!;
  await noisyImg.decode();

  const denoiser = await Denoiser.create();
  const denoised = await denoiser.denoise(noisyImg);

  const canvas = document.querySelector<HTMLCanvasElement>('#denoised')!;
  canvas.getContext('2d')!.putImageData(denoised!, 0, 0);

  return denoiser;
}

// --- page plumbing ---

import pageSource from './main.ts?raw';
import { ensureWebGPU, demoFooter } from '../../_shared/chrome';

const statusEl = document.querySelector<HTMLElement>('#status')!;
const snippetEl = document.querySelector<HTMLElement>('#snippet')!;
const noisyImgEl = document.querySelector<HTMLImageElement>('#noisy-img')!;
const noisyCanvas = document.querySelector<HTMLCanvasElement>('#noisy')!;

// Show the exact code above, verbatim, so this page can never drift from reality.
snippetEl.textContent = pageSource.split('// --- page plumbing ---')[0].trim();

// Paint the noisy source into its own canvas so both panels are canvases.
noisyImgEl.decode().then(() => {
  noisyCanvas.width = noisyImgEl.naturalWidth;
  noisyCanvas.height = noisyImgEl.naturalHeight;
  noisyCanvas.getContext('2d')!.drawImage(noisyImgEl, 0, 0);
});

async function main() {
  // Shared demo chrome: friendly full-page banner + bail if WebGPU is unavailable.
  if (!(await ensureWebGPU())) {
    statusEl.textContent = '';
    return;
  }
  statusEl.textContent = 'Fetching model + creating WebGPU device...';
  const t0 = performance.now();
  const denoiser = await run();
  const totalMs = performance.now() - t0;
  const denoiseMs = denoiser.stats?.totalMs ?? totalMs;
  const setupMs = Math.max(0, totalMs - denoiseMs);
  statusEl.textContent =
    `model + device ready in ${setupMs.toFixed(0)} ms -> denoised in ${denoiseMs.toFixed(0)} ms ` +
    `(${totalMs.toFixed(0)} ms total)`;
}

demoFooter('hello-world');
main();
