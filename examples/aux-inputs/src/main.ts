import { Denoiser } from 'denoiser';
import { ensureWebGPU, demoFooter } from '../../_shared/chrome';

// ---- scene ------------------------------------------------------------------
// Single scene (LDraw Eiffel Tower, 512x512) is enough: the point is the aux
// contract, not scene variety. Assets copied from examples/gallery/public/scenes/eiffel/.
// (Measured: the eiffel scene's tower/rooftop silhouette shows corruption effects
// more clearly than the flatter `spheres` scene — see verification notes.)

const SCENE_DIR = './scenes/eiffel/';
const COLOR_FILE = 'spp4.png';
const ALBEDO_FILE = 'albedo.png';
const NORMAL_FILE = 'normal.png';

// ---- corruption types ---------------------------------------------------------

type CorruptionId = 'none' | 'env-normals' | 'srgb-albedo' | 'missing-aux';

interface CorruptionDef {
  id: CorruptionId;
  label: string;
  blurb: string;
}

const CORRUPTIONS: CorruptionDef[] = [
  {
    id: 'none',
    label: 'Correct aux (baseline)',
    blurb: 'No corruption — both panels show the same correct-aux denoise. Click a bug below to break it.',
  },
  {
    id: 'env-normals',
    label: 'Env-gradient normals',
    blurb:
      'The classic "MRT pass with background on" bug: instead of OIDN\'s normal=0 convention for ' +
      'env/miss pixels, the background left a smooth interpolated screen-space gradient in the normal G-buffer. ' +
      'Honest note: on a smooth open sky like this the network leans on color and barely reacts (watch the tiny Δ) — ' +
      'the same violation bites hard where silhouette edges cross the environment.',
  },
  {
    id: 'srgb-albedo',
    label: 'sRGB-encoded albedo',
    blurb:
      'Albedo must be linear [0,1] bytes. Here the sRGB transfer curve was applied before handing the ' +
      'bytes to the network — as if a gamma-encoded texture/screenshot were fed in where linear is required.',
  },
  {
    id: 'missing-aux',
    label: 'Missing aux',
    blurb: 'No albedo or normal passed at all — the plain 3-channel color-only model.',
  },
];

// ---- DOM ------------------------------------------------------------------

const viewColorEl = document.querySelector<HTMLCanvasElement>('#view-color')!;
const viewAlbedoEl = document.querySelector<HTMLCanvasElement>('#view-albedo')!;
const viewNormalEl = document.querySelector<HTMLCanvasElement>('#view-normal')!;
const viewNormalDecodedEl = document.querySelector<HTMLCanvasElement>('#view-normal-decoded')!;

const corruptionControlEl = document.querySelector<HTMLElement>('#corruption-control')!;
const corruptionBlurbEl = document.querySelector<HTMLElement>('#corruption-blurb')!;
const corruptedPlanesEl = document.querySelector<HTMLElement>('#corrupted-planes')!;
const baselineCanvas = document.querySelector<HTMLCanvasElement>('#canvas-baseline')!;
const activeCanvas = document.querySelector<HTMLCanvasElement>('#canvas-active')!;
const statusEl = document.querySelector<HTMLElement>('#status')!;
const loadingEl = document.querySelector<HTMLElement>('#loading')!;

// ---- image loading ----------------------------------------------------------

async function loadImage(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

function toImageData(img: HTMLImageElement): ImageData {
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height);
}

function cloneImageData(id: ImageData): ImageData {
  return new ImageData(new Uint8ClampedArray(id.data), id.width, id.height);
}

function paint(canvas: HTMLCanvasElement, data: ImageData): void {
  canvas.width = data.width;
  canvas.height = data.height;
  canvas.getContext('2d')!.putImageData(data, 0, 0);
}

// ---- label burning ----------------------------------------------------------

function burnLabel(canvas: HTMLCanvasElement, text: string, corner: 'left' | 'right' = 'left'): void {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const h = canvas.height;
  ctx.save();
  ctx.font = `${Math.max(12, Math.round(w / 34))}px system-ui, sans-serif`;
  ctx.textBaseline = 'bottom';
  const metrics = ctx.measureText(text);
  const padX = 8;
  const padY = 6;
  const boxW = metrics.width + padX * 2;
  const boxH = parseInt(ctx.font, 10) + padY * 2;
  const x = corner === 'left' ? 10 : w - boxW - 10;
  const y = h - 10 - boxH;
  ctx.fillStyle = 'rgba(11,14,20,0.72)';
  ctx.fillRect(x, y, boxW, boxH);
  ctx.fillStyle = '#e6edf3';
  ctx.textAlign = 'left';
  ctx.fillText(text, x + padX, y + boxH - padY);
  ctx.restore();
}

// ---- normal decode debug view ------------------------------------------------

/** Colors each pixel by the length of the decoded [-1,1] normal: blue = 0 (correct
 *  env/miss convention), green = 1 (valid unit normal), red = broken/over-length. */
function normalLengthHeatmap(normalData: ImageData): ImageData {
  const { width, height, data } = normalData;
  const out = new ImageData(width, height);
  for (let i = 0; i < data.length; i += 4) {
    const nx = (data[i] / 255) * 2 - 1;
    const ny = (data[i + 1] / 255) * 2 - 1;
    const nz = (data[i + 2] / 255) * 2 - 1;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const t = Math.min(1, len);
    out.data[i] = len > 1.05 ? 255 : 0; // red past unit length
    out.data[i + 1] = Math.round(255 * t); // green rises to unit length
    out.data[i + 2] = Math.round(255 * (1 - t)); // blue at zero (env)
    out.data[i + 3] = 255;
  }
  return out;
}

// ---- corruption builders ------------------------------------------------------

/** True where the correct normal buffer is exactly OIDN's env/miss convention
 *  (encoded 0 -> byte 128 gray). */
function computeEnvMask(normalData: ImageData): Uint8Array {
  const { width, height, data } = normalData;
  const mask = new Uint8Array(width * height);
  for (let p = 0, i = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    mask[p] = Math.abs(r - 128) <= 2 && Math.abs(g - 128) <= 2 && Math.abs(b - 128) <= 2 ? 1 : 0;
  }
  return mask;
}

/** Reproduces the "MRT pass with background on" bug: env pixels get a smooth
 *  screen-space gradient instead of the flat normal=0 (0.5 gray) OIDN expects. */
function buildEnvGradientNormal(normalData: ImageData, envMask: Uint8Array): ImageData {
  const { width, height } = normalData;
  const out = cloneImageData(normalData);
  for (let y = 0, p = 0; y < height; y++) {
    for (let x = 0; x < width; x++, p++) {
      if (!envMask[p]) continue;
      const i = p * 4;
      const u = x / (width - 1);
      const v = y / (height - 1);
      out.data[i] = Math.round(255 * u);
      out.data[i + 1] = Math.round(255 * (1 - v));
      out.data[i + 2] = Math.round(255 * (0.3 + 0.4 * v));
      out.data[i + 3] = 255;
    }
  }
  return out;
}

function srgbEncodeByte(byte: number): number {
  const c = byte / 255;
  const s = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(s * 255)));
}

/** Simulates feeding gamma-encoded (sRGB) albedo bytes where linear [0,1] is required. */
function buildSrgbAlbedo(albedoData: ImageData): ImageData {
  const { width, height, data } = albedoData;
  const out = new ImageData(width, height);
  for (let i = 0; i < data.length; i += 4) {
    out.data[i] = srgbEncodeByte(data[i]);
    out.data[i + 1] = srgbEncodeByte(data[i + 1]);
    out.data[i + 2] = srgbEncodeByte(data[i + 2]);
    out.data[i + 3] = 255;
  }
  return out;
}

// ---- diff metric --------------------------------------------------------------

/** Mean absolute per-channel RGB difference between two same-sized ImageData. */
function meanAbsDiff(a: ImageData, b: ImageData): number {
  let sum = 0;
  let n = 0;
  const da = a.data;
  const db = b.data;
  for (let i = 0; i < da.length; i += 4) {
    sum += Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2]);
    n += 3;
  }
  return sum / n;
}

// ---- state ------------------------------------------------------------------

let denoiser: Denoiser;
let colorData: ImageData;
let albedoData: ImageData;
let normalData: ImageData;
let baselineResult: ImageData;
let baselineMs = 0;
let activeCorruption: CorruptionId = 'none';

const corruptedAssets = new Map<CorruptionId, { albedo?: ImageData; normal?: ImageData }>();
const resultCache = new Map<CorruptionId, { imageData: ImageData; ms: number }>();

// Serialize denoise calls so rapidly clicking the segmented control never
// overlaps two runs on the shared engine/device.
let runChain: Promise<void> = Promise.resolve();
function serialize(fn: () => Promise<void>): Promise<void> {
  const next = runChain.then(fn, fn);
  runChain = next;
  return next;
}

// ---- section 1: "what the network sees" --------------------------------------

function paintWhatNetworkSees(): void {
  paint(viewColorEl, colorData);
  burnLabel(viewColorEl, '4 spp noisy');

  paint(viewAlbedoEl, albedoData);
  burnLabel(viewAlbedoEl, 'albedo (linear)');

  paint(viewNormalEl, normalData);
  burnLabel(viewNormalEl, 'normal (encoded)');

  paint(viewNormalDecodedEl, normalLengthHeatmap(normalData));
  burnLabel(viewNormalDecodedEl, 'decoded length');
}

// ---- section 2: "break it and see" --------------------------------------------

function renderCorruptionControl(): void {
  corruptionControlEl.innerHTML = '';
  for (const c of CORRUPTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = c.label;
    btn.setAttribute('aria-pressed', String(c.id === activeCorruption));
    btn.addEventListener('click', () => {
      if (c.id === activeCorruption) return;
      void selectCorruption(c.id);
    });
    corruptionControlEl.appendChild(btn);
  }
  corruptionBlurbEl.textContent = CORRUPTIONS.find((c) => c.id === activeCorruption)!.blurb;
}

function renderCorruptedPlanes(id: CorruptionId): void {
  corruptedPlanesEl.innerHTML = '';
  if (id === 'none') {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'no corruption active — inputs are identical to the baseline above';
    corruptedPlanesEl.appendChild(p);
    return;
  }
  if (id === 'missing-aux') {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'albedo and normal are not passed to denoise() at all — nothing to render';
    corruptedPlanesEl.appendChild(p);
    return;
  }
  const assets = corruptedAssets.get(id)!;
  const entries: [string, ImageData | undefined][] =
    id === 'env-normals' ? [['normal (corrupted)', assets.normal]] : [['albedo (corrupted)', assets.albedo]];
  for (const [label, data] of entries) {
    if (!data) continue;
    const fig = document.createElement('figure');
    const canvas = document.createElement('canvas');
    fig.appendChild(canvas);
    const figcaption = document.createElement('figcaption');
    figcaption.innerHTML = `<p class="t">${label}</p>`;
    fig.appendChild(figcaption);
    corruptedPlanesEl.appendChild(fig);
    paint(canvas, data);
    burnLabel(canvas, label);
  }
}

async function runCorruption(id: CorruptionId): Promise<{ imageData: ImageData; ms: number }> {
  const cached = resultCache.get(id);
  if (cached) return cached;

  if (id === 'none') {
    const entry = { imageData: baselineResult, ms: baselineMs };
    resultCache.set(id, entry);
    return entry;
  }

  const assets = corruptedAssets.get(id) ?? {};
  const useAlbedo = id === 'missing-aux' ? undefined : (assets.albedo ?? albedoData);
  const useNormal = id === 'missing-aux' ? undefined : (assets.normal ?? normalData);

  const out = await denoiser.denoise(colorData, { albedo: useAlbedo, normal: useNormal });
  const ms = denoiser.stats?.totalMs ?? 0;
  const entry = { imageData: out!, ms };
  resultCache.set(id, entry);
  return entry;
}

async function selectCorruption(id: CorruptionId): Promise<void> {
  activeCorruption = id;
  renderCorruptionControl();
  renderCorruptedPlanes(id);

  await serialize(async () => {
    loadingEl.textContent = 'denoising...';
    const { imageData, ms } = await runCorruption(id);

    paint(activeCanvas, imageData);
    burnLabel(activeCanvas, CORRUPTIONS.find((c) => c.id === id)!.label, 'right');

    const diff = meanAbsDiff(baselineResult, imageData);
    const def = CORRUPTIONS.find((c) => c.id === id)!;
    const name = denoiser.modelName ?? 'unknown';
    const baselineClass = id === 'none' ? ' is-baseline' : '';
    statusEl.innerHTML =
      `model <span class="model">${name}</span> · ${ms.toFixed(1)} ms · corruption: ` +
      `<span class="corruption-name${baselineClass}">${def.label}</span> · ` +
      `mean |Δ| vs correct-aux: ${diff.toFixed(2)}`;
    loadingEl.textContent = '';
  });
}

// ---- boot -------------------------------------------------------------------------

async function main(): Promise<void> {
  if (!(await ensureWebGPU())) return;

  loadingEl.textContent = 'loading scene assets...';
  const [colorImg, albedoImg, normalImg] = await Promise.all([
    loadImage(`${SCENE_DIR}${COLOR_FILE}`),
    loadImage(`${SCENE_DIR}${ALBEDO_FILE}`),
    loadImage(`${SCENE_DIR}${NORMAL_FILE}`),
  ]);
  colorData = toImageData(colorImg);
  albedoData = toImageData(albedoImg);
  normalData = toImageData(normalImg);

  paintWhatNetworkSees();

  // Corruptions are computed once from the correct assets and cached — no extra fetches.
  const envMask = computeEnvMask(normalData);
  corruptedAssets.set('env-normals', { normal: buildEnvGradientNormal(normalData, envMask) });
  corruptedAssets.set('srgb-albedo', { albedo: buildSrgbAlbedo(albedoData) });
  corruptedAssets.set('missing-aux', {});
  corruptedAssets.set('none', {});

  loadingEl.textContent = 'fetching model + creating WebGPU device...';
  denoiser = await Denoiser.create();

  loadingEl.textContent = 'denoising (correct-aux baseline)...';
  const baseline = await denoiser.denoise(colorData, { albedo: albedoData, normal: normalData });
  baselineResult = baseline!;
  baselineMs = denoiser.stats?.totalMs ?? 0;
  paint(baselineCanvas, baselineResult);
  burnLabel(baselineCanvas, 'correct aux', 'left');

  renderCorruptionControl();
  await selectCorruption('none');

  loadingEl.textContent = '';

  // Automation/debugging hook (headless verification, console poking) — same
  // pattern as examples/ldraw-eiffel and examples/three-pathtracer-webgpu.
  (window as unknown as Record<string, unknown>).__app = {
    denoiser,
    selectCorruption,
    getBaseline: () => baselineResult,
    getResult: (id: CorruptionId) => resultCache.get(id)?.imageData,
    meanAbsDiff,
  };
}

demoFooter('aux-inputs');
main().catch((err) => {
  console.error(err);
  statusEl.textContent = `error: ${(err as Error).message}`;
  loadingEl.textContent = '';
});
