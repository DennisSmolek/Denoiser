import modelUrl from '../../../packages/denoiser/models/rt_ldr_small.onnx?url';
import auxModelUrl from '../../../packages/denoiser/models/rt_ldr_alb_nrm_small.onnx?url';
import { DenoiseSession, rgbaToNCHW, nchwToRGBA } from './denoiseSession';

const SIZE = 256;
const BIG_W = 640;
const BIG_H = 384;

const noisyCanvas = document.querySelector<HTMLCanvasElement>('#noisy')!;
const cleanCanvas = document.querySelector<HTMLCanvasElement>('#clean')!;
const noisyBig = document.querySelector<HTMLCanvasElement>('#noisyBig')!;
const cleanBig = document.querySelector<HTMLCanvasElement>('#cleanBig')!;
const auxColor = document.querySelector<HTMLCanvasElement>('#auxColor')!;
const auxAlbedo = document.querySelector<HTMLCanvasElement>('#auxAlbedo')!;
const auxNormal = document.querySelector<HTMLCanvasElement>('#auxNormal')!;
const auxClean = document.querySelector<HTMLCanvasElement>('#auxClean')!;
const runBtn = document.querySelector<HTMLButtonElement>('#run')!;
const benchBtn = document.querySelector<HTMLButtonElement>('#bench')!;
const runTiledBtn = document.querySelector<HTMLButtonElement>('#runTiled')!;
const runAuxBtn = document.querySelector<HTMLButtonElement>('#runAux')!;
const status = document.querySelector<HTMLPreElement>('#status')!;

function log(msg: string) {
  status.textContent += msg + '\n';
  console.log(msg);
}

/** Draw a smooth synthetic scene; optionally add gaussian noise (stand-in for MC render noise). */
function makeScene(w = SIZE, h = SIZE, addNoise = true): ImageData {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  // smooth gradient background
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, '#2b6cb0');
  g.addColorStop(1, '#f6ad55');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  // a few smooth shapes, scaled to the canvas
  for (const [fx, fy, fr, col] of [
    [0.3, 0.35, 0.2, '#e53e3e'],
    [0.7, 0.66, 0.24, '#38a169'],
    [0.78, 0.23, 0.12, '#d69e2e'],
  ] as const) {
    ctx.fillStyle = col as string;
    ctx.beginPath();
    ctx.arc((fx as number) * w, (fy as number) * h, (fr as number) * Math.min(w, h), 0, Math.PI * 2);
    ctx.fill();
  }
  const img = ctx.getImageData(0, 0, w, h);
  if (addNoise) {
    const sigma = 40;
    for (let i = 0; i < img.data.length; i += 4) {
      for (let ch = 0; ch < 3; ch++) {
        const n = (Math.random() + Math.random() + Math.random() - 1.5) * sigma;
        img.data[i + ch] = Math.max(0, Math.min(255, img.data[i + ch] + n));
      }
    }
  }
  return img;
}

const makeNoisyImage = (w = SIZE, h = SIZE) => makeScene(w, h, true);

/** A flat synthetic normal map (encoded (0,0,1) = (128,128,255)) — plumbing stand-in. */
function makeNormalImage(w = SIZE, h = SIZE): ImageData {
  const img = new ImageData(w, h);
  for (let i = 0; i < img.data.length; i += 4) {
    img.data[i] = 128;
    img.data[i + 1] = 128;
    img.data[i + 2] = 255;
    img.data[i + 3] = 255;
  }
  return img;
}

async function run(session: DenoiseSession) {
  status.textContent = '';
  const noisy = makeNoisyImage();
  noisyCanvas.getContext('2d')!.putImageData(noisy, 0, 0);

  log('running full-GPU path (WGSL pre/post + WebGPU EP, IO-bound)...');
  const t0 = performance.now();
  const rgbaOut = await session.denoiseTileGPU(noisy.data);
  const dt = performance.now() - t0;
  log(`done in ${dt.toFixed(1)} ms`);

  // TS 5.7+ lib.dom typings narrowed ImageData's data param to Uint8ClampedArray<ArrayBuffer>;
  // this example never uses SharedArrayBuffer, so this cast is safe.
  cleanCanvas.getContext('2d')!.putImageData(new ImageData(rgbaOut as Uint8ClampedArray<ArrayBuffer>, SIZE, SIZE), 0, 0);

  // correctness A/B: compare the all-GPU result to the JS-preprocessed path
  const nchw = rgbaToNCHW(noisy.data, SIZE, SIZE, session.channels);
  const jsOut = nchwToRGBA(await session.denoiseTile(nchw), SIZE, SIZE);
  let maxDiff = 0;
  for (let i = 0; i < rgbaOut.length; i++) maxDiff = Math.max(maxDiff, Math.abs(rgbaOut[i] - jsOut[i]));
  log(`GPU vs JS path max pixel diff: ${maxDiff} (expect 0–1 from rounding)`);
}

async function benchmark(session: DenoiseSession) {
  const noisy = makeNoisyImage();
  // warm up (first run compiles WGSL shaders + ORT pipelines)
  await session.denoiseTileGPU(noisy.data);
  const N = 50;
  const times: number[] = [];
  for (let i = 0; i < N; i++) {
    const t0 = performance.now();
    await session.denoiseTileGPU(noisy.data);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const avg = times.reduce((a, b) => a + b, 0) / N;
  log(`bench ×${N} (full-GPU path): min ${times[0].toFixed(1)}  avg ${avg.toFixed(1)}  ` +
      `median ${times[N >> 1].toFixed(1)}  max ${times[N - 1].toFixed(1)} ms ` +
      `(only final pixel readback on CPU)`);
}

async function runTiled(session: DenoiseSession) {
  const noisy = makeNoisyImage(BIG_W, BIG_H);
  noisyBig.getContext('2d')!.putImageData(noisy, 0, 0);
  const grid = session.tileGrid(BIG_W, BIG_H);
  log(`tiling ${BIG_W}×${BIG_H}: ${grid.tilesX}×${grid.tilesY} = ${grid.total} tiles (256², overlap 32)`);
  const t0 = performance.now();
  const out = await session.denoiseImage(noisy.data, BIG_W, BIG_H);
  const dt = performance.now() - t0;
  log(`tiled denoise done in ${dt.toFixed(1)} ms (${(dt / grid.total).toFixed(1)} ms/tile)`);
  // See ArrayBuffer-vs-ArrayBufferLike note above.
  cleanBig.getContext('2d')!.putImageData(new ImageData(out as Uint8ClampedArray<ArrayBuffer>, BIG_W, BIG_H), 0, 0);
}

async function runAux(session: DenoiseSession) {
  // color = noisy scene; albedo = the clean base scene; normal = flat synthetic
  const color = makeScene(SIZE, SIZE, true);
  const albedo = makeScene(SIZE, SIZE, false);
  const normal = makeNormalImage(SIZE, SIZE);
  auxColor.getContext('2d')!.putImageData(color, 0, 0);
  auxAlbedo.getContext('2d')!.putImageData(albedo, 0, 0);
  auxNormal.getContext('2d')!.putImageData(normal, 0, 0);

  log('running 9ch aux path (color+albedo+normal concat in WGSL)...');
  const t0 = performance.now();
  const out = await session.denoiseAuxTile(color.data, albedo.data, normal.data);
  log(`aux denoise done in ${(performance.now() - t0).toFixed(1)} ms`);
  // See ArrayBuffer-vs-ArrayBufferLike note above.
  auxClean.getContext('2d')!.putImageData(new ImageData(out as Uint8ClampedArray<ArrayBuffer>, SIZE, SIZE), 0, 0);
}

async function main() {
  if (!('gpu' in navigator)) {
    log('ERROR: WebGPU not available in this browser.');
    return;
  }
  log('loading model: ' + modelUrl);
  const session = new DenoiseSession();
  await session.init({ modelUrl, channels: 3, size: SIZE });
  log(`session ready. input=${session.inputName} output=${session.outputName}`);
  log(`shared GPUDevice acquired: ${session.device ? 'yes' : 'no'}`);

  runBtn.disabled = false;
  benchBtn.disabled = false;
  runTiledBtn.disabled = false;
  runBtn.addEventListener('click', () => run(session).catch((e) => log('ERROR: ' + e.message)));
  benchBtn.addEventListener('click', () => benchmark(session).catch((e) => log('ERROR: ' + e.message)));
  runTiledBtn.addEventListener('click', () => runTiled(session).catch((e) => log('ERROR: ' + e.message)));
  await run(session); // run once on load
  await runTiled(session); // and the tiled path once on load

  // second session for the 9-channel aux model
  log('loading 9ch aux model: ' + auxModelUrl);
  const auxSession = new DenoiseSession();
  await auxSession.init({ modelUrl: auxModelUrl, channels: 9, size: SIZE });
  log(`aux session ready. input=${auxSession.inputName} output=${auxSession.outputName}`);
  runAuxBtn.disabled = false;
  runAuxBtn.addEventListener('click', () => runAux(auxSession).catch((e) => log('ERROR: ' + e.message)));
  await runAux(auxSession);
}

main().catch((e) => log('ERROR: ' + e.message));
