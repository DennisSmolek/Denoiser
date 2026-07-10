import { Denoiser } from 'denoiser';
import { ensureWebGPU, demoFooter } from '../../_shared/chrome';

// ---- manifest -------------------------------------------------------------

interface SceneManifest {
  id: string;
  title: string;
  width: number;
  height: number;
  spp: number[];
  color: Record<string, string>; // spp (as string) -> filename, relative to the scene folder
  albedo?: string;
  normal?: string;
  reference?: string;
}

interface Manifest {
  scenes: SceneManifest[];
}

// ---- DOM ------------------------------------------------------------------

const scenesEl = document.querySelector<HTMLElement>('#scenes')!;
const sppControlEl = document.querySelector<HTMLElement>('#spp-control')!;
const auxToggleEl = document.querySelector<HTMLInputElement>('#aux-toggle')!;
const auxHintEl = document.querySelector<HTMLElement>('#aux-hint')!;
const refBtnEl = document.querySelector<HTMLButtonElement>('#ref-btn')!;
const compareEl = document.querySelector<HTMLElement>('#compare')!;
const dividerEl = document.querySelector<HTMLElement>('#divider')!;
const noisyCanvas = document.querySelector<HTMLCanvasElement>('#canvas-noisy')!;
const denoisedCanvas = document.querySelector<HTMLCanvasElement>('#canvas-denoised')!;
const referenceCanvas = document.querySelector<HTMLCanvasElement>('#canvas-reference')!;
const statusEl = document.querySelector<HTMLElement>('#status')!;
const loadingEl = document.querySelector<HTMLElement>('#loading')!;

const noisyCtx = noisyCanvas.getContext('2d')!;
const denoisedCtx = denoisedCanvas.getContext('2d')!;
const referenceCtx = referenceCanvas.getContext('2d')!;

// ---- state ------------------------------------------------------------------

let manifest: Manifest;
let scene: SceneManifest;
let spp = 0;
let auxOn = false;
let denoiser: Denoiser;

// Small image cache keyed by resolved URL so switching spp/scene back and forth
// doesn't redecode PNGs it has already fetched.
const imageCache = new Map<string, HTMLImageElement>();
async function loadImage(url: string): Promise<HTMLImageElement> {
  const cached = imageCache.get(url);
  if (cached) return cached;
  const img = new Image();
  img.src = url;
  await img.decode();
  imageCache.set(url, img);
  return img;
}

function sceneUrl(s: SceneManifest, file: string): string {
  return `./scenes/${s.id}/${file}`;
}

// All denoise calls share one engine + one set of GPU buffers sized to the
// image; serialize them so a rapid spp/aux flip never overlaps two runs.
let runChain: Promise<void> = Promise.resolve();
function serialize(fn: () => Promise<void>): void {
  runChain = runChain.then(fn, fn);
}

// ---- label burning ----------------------------------------------------------

function burnLabel(ctx: CanvasRenderingContext2D, text: string, corner: 'left' | 'right'): void {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
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

// ---- rendering ---------------------------------------------------------------

async function paintNoisy(): Promise<void> {
  const url = sceneUrl(scene, scene.color[String(spp)]);
  const img = await loadImage(url);
  noisyCanvas.width = scene.width;
  noisyCanvas.height = scene.height;
  noisyCtx.drawImage(img, 0, 0, scene.width, scene.height);
  burnLabel(noisyCtx, `noisy · ${spp} spp`, 'left');
}

async function runDenoise(): Promise<void> {
  loadingEl.textContent = 'denoising...';
  const colorUrl = sceneUrl(scene, scene.color[String(spp)]);
  const colorImg = await loadImage(colorUrl);

  const hasAux = !!(scene.albedo && scene.normal);
  const useAux = auxOn && hasAux;
  let albedoImg: HTMLImageElement | undefined;
  let normalImg: HTMLImageElement | undefined;
  if (useAux) {
    [albedoImg, normalImg] = await Promise.all([
      loadImage(sceneUrl(scene, scene.albedo!)),
      loadImage(sceneUrl(scene, scene.normal!)),
    ]);
  }

  const out = await denoiser.denoise(colorImg, { albedo: albedoImg, normal: normalImg });
  if (!out) {
    loadingEl.textContent = '';
    return; // aborted mid-run (scene/quality changed elsewhere)
  }

  denoisedCanvas.width = scene.width;
  denoisedCanvas.height = scene.height;
  denoisedCtx.putImageData(out, 0, 0);
  burnLabel(denoisedCtx, 'denoised', 'right');

  const name = denoiser.modelName ?? 'unknown';
  const channels = useAux ? 9 : 3;
  const stats = denoiser.stats;
  const ms = stats?.totalMs.toFixed(1) ?? '?';
  const splitNote = useAux ? ' (splitAux workaround active)' : '';
  statusEl.innerHTML =
    `model <span class="model">${name}</span> · ${channels}ch${splitNote} · ` +
    `denoised in ${ms} ms · ${scene.width}×${scene.height} · ${spp} spp`;
  loadingEl.textContent = '';
}

function refresh(): void {
  serialize(async () => {
    await paintNoisy();
    await runDenoise();
  });
}

// ---- scene picker -------------------------------------------------------------

function renderScenePicker(): void {
  scenesEl.innerHTML = '';
  for (const s of manifest.scenes) {
    const btn = document.createElement('button');
    btn.className = 'scene-card';
    btn.setAttribute('aria-pressed', String(s.id === scene.id));
    btn.innerHTML = `<p class="t">${escapeHtml(s.title)}</p><p class="d">${s.width}×${s.height} · spp: ${s.spp.join(', ')}</p>`;
    btn.addEventListener('click', () => selectScene(s));
    scenesEl.appendChild(btn);
  }
}

function selectScene(s: SceneManifest): void {
  scene = s;
  spp = s.spp[s.spp.length - 1]; // default to the highest available spp
  const hasAux = !!(s.albedo && s.normal);
  auxToggleEl.disabled = !hasAux;
  auxToggleEl.checked = hasAux; // aux on by default when available — that's the point of the demo
  auxOn = hasAux;
  auxHintEl.textContent = hasAux ? 'color + albedo + normal, cleanAux model' : 'this scene has no aux buffers';
  refBtnEl.disabled = !s.reference;
  refBtnEl.textContent = s.reference ? 'hold to peek converged frame' : 'no reference for this scene';

  renderScenePicker();
  renderSppControl();
  refresh();
}

// ---- spp segmented control -----------------------------------------------------

function renderSppControl(): void {
  sppControlEl.innerHTML = '';
  for (const s of scene.spp) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = String(s);
    btn.setAttribute('aria-pressed', String(s === spp));
    btn.addEventListener('click', () => {
      if (s === spp) return;
      spp = s;
      renderSppControl();
      refresh();
    });
    sppControlEl.appendChild(btn);
  }
}

// ---- aux toggle -----------------------------------------------------------------

auxToggleEl.addEventListener('change', () => {
  auxOn = auxToggleEl.checked;
  refresh();
});

// ---- reference peek (hold button) ------------------------------------------------

async function showReference(): Promise<void> {
  if (!scene.reference) return;
  const img = await loadImage(sceneUrl(scene, scene.reference));
  referenceCanvas.width = scene.width;
  referenceCanvas.height = scene.height;
  referenceCtx.drawImage(img, 0, 0, scene.width, scene.height);
  burnLabel(referenceCtx, 'reference · converged', 'left');
  referenceCanvas.style.display = 'block';
}
function hideReference(): void {
  referenceCanvas.style.display = 'none';
}
refBtnEl.addEventListener('pointerdown', (e) => {
  if (refBtnEl.disabled) return;
  refBtnEl.setPointerCapture(e.pointerId);
  void showReference();
});
refBtnEl.addEventListener('pointerup', hideReference);
refBtnEl.addEventListener('pointercancel', hideReference);
refBtnEl.addEventListener('lostpointercapture', hideReference);

// ---- before/after divider (hand-rolled, pointer events) --------------------------

let dividerPct = 50;
function setDivider(pct: number): void {
  dividerPct = Math.min(100, Math.max(0, pct));
  denoisedCanvas.style.clipPath = `inset(0 0 0 ${dividerPct}%)`;
  dividerEl.style.left = `${dividerPct}%`;
}
setDivider(50);

let dragging = false;
function pctFromEvent(e: PointerEvent): number {
  const rect = compareEl.getBoundingClientRect();
  return ((e.clientX - rect.left) / rect.width) * 100;
}
compareEl.addEventListener('pointerdown', (e) => {
  dragging = true;
  compareEl.setPointerCapture(e.pointerId);
  setDivider(pctFromEvent(e));
});
compareEl.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  setDivider(pctFromEvent(e));
});
function endDrag(): void {
  dragging = false;
}
compareEl.addEventListener('pointerup', endDrag);
compareEl.addEventListener('pointercancel', endDrag);

// ---- boot -------------------------------------------------------------------------

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

async function main(): Promise<void> {
  if (!(await ensureWebGPU())) return;

  loadingEl.textContent = 'loading scene manifest...';
  const res = await fetch('./scenes/manifest.json');
  manifest = (await res.json()) as Manifest;
  if (!manifest.scenes.length) {
    statusEl.textContent = 'no scenes in manifest.json';
    return;
  }

  loadingEl.textContent = 'fetching model + creating WebGPU device...';
  denoiser = await Denoiser.create();

  selectScene(manifest.scenes[0]);
}

demoFooter('gallery');
main().catch((err) => {
  console.error(err);
  statusEl.textContent = `error: ${(err as Error).message}`;
  loadingEl.textContent = '';
});
