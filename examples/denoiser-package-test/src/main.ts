import { Denoiser } from 'denoiser';

const SIZE = 384; // not a multiple of 256 -> exercises tiling + padding via the package
const noisyCanvas = document.querySelector<HTMLCanvasElement>('#noisy')!;
const cleanCanvas = document.querySelector<HTMLCanvasElement>('#clean')!;
const runBtn = document.querySelector<HTMLButtonElement>('#run')!;
const status = document.querySelector<HTMLPreElement>('#status')!;

const log = (m: string) => { status.textContent += m + '\n'; console.log(m); };

function makeNoisy(): ImageData {
  const c = document.createElement('canvas');
  c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext('2d')!;
  const g = ctx.createLinearGradient(0, 0, SIZE, SIZE);
  g.addColorStop(0, '#2b6cb0'); g.addColorStop(1, '#f6ad55');
  ctx.fillStyle = g; ctx.fillRect(0, 0, SIZE, SIZE);
  for (const [x, y, r, col] of [[120, 130, 70, '#e53e3e'], [260, 250, 80, '#38a169']] as const) {
    ctx.fillStyle = col as string;
    ctx.beginPath(); ctx.arc(x as number, y as number, r as number, 0, Math.PI * 2); ctx.fill();
  }
  const img = ctx.getImageData(0, 0, SIZE, SIZE);
  for (let i = 0; i < img.data.length; i += 4)
    for (let ch = 0; ch < 3; ch++) {
      const n = (Math.random() + Math.random() + Math.random() - 1.5) * 40;
      img.data[i + ch] = Math.max(0, Math.min(255, img.data[i + ch] + n));
    }
  return img;
}

async function run(denoiser: Denoiser) {
  const noisy = makeNoisy();
  noisyCanvas.getContext('2d')!.putImageData(noisy, 0, 0);
  log('executing denoiser.denoise(noisy)...');
  const t0 = performance.now();
  const out = await denoiser.denoise(noisy);
  if (out) cleanCanvas.getContext('2d')!.putImageData(out, 0, 0);
  log(`done in ${(performance.now() - t0).toFixed(1)} ms (model: rt_ldr_small)`);
}

async function main() {
  if (!('gpu' in navigator)) { log('ERROR: WebGPU not available.'); return; }
  log('creating Denoiser (loads model + device)...');
  const denoiser = await Denoiser.create({ quality: 'fast', weightsUrl: '/models' });
  await run(denoiser);
  log(`shared GPUDevice exposed: ${denoiser.device ? 'yes' : 'no'}`);
  runBtn.disabled = false;
  runBtn.addEventListener('click', () => run(denoiser).catch((e) => log('ERROR: ' + e.message)));
}

main().catch((e) => log('ERROR: ' + e.message));
