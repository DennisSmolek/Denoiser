// Phase B2 demo #8 — the "three packages, one GPUDevice" pipeline.
//
//   three.js (render @ low res) → denoiser (clean) → @pmndrs/upscaler (FSR1) → canvas
//
// All three libraries share the SINGLE GPUDevice that the denoiser (ORT) creates.
// Every stage hands a GPUTexture straight to the next — no CPU readback in the
// chain. The denoiser writes its result into a three StorageTexture; the upscaler
// consumes that three texture directly (it resolves the backing GPUTexture via the
// exact same `renderer.backend.get(tex).texture` handle three itself uses), and its
// output is another three texture we present with a fullscreen quad.
//
// Composition note: the denoiser is a discrete, stills/cadence-oriented pass, so we
// use the upscaler's FSR1 *spatial* path (per-frame, stateless, no motion vectors) —
// the honest fit. The library also ships an FSR2/3-style *temporal* path, which wants
// per-frame depth + velocity + camera jitter; that's a different (motion-first)
// integration and is out of scope for a denoised-stills pipeline.
import * as THREE from 'three/webgpu';
import {
  Fn, texture, uv, vec2, vec3, vec4, float, uniform, hash, mix, step, screenCoordinate,
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Denoiser } from 'denoiser';
import { Upscaler } from '@pmndrs/upscaler';
import { ensureWebGPU, demoFooter } from '../../_shared/chrome';

// ---- resolution --------------------------------------------------------------
const RW = 640, RH = 360;   // render (denoise) resolution
const DW = 1280, DH = 720;  // display (canvas) resolution — 2.0× spatial upscale

const statusEl = document.querySelector<HTMLParagraphElement>('#status')!;
const log = (m: string, err = false) => {
  statusEl.textContent = m; statusEl.classList.toggle('err', err); console.log(m);
};

// Patch requestDevice to request the adapter's MAX limits + all features BEFORE any
// device is created. ORT creates the shared device first and otherwise asks for a
// minimal one; three's renderer and the upscaler's compute passes then share that
// single device, so it must be capable enough for all three. (onnxruntime #26107.)
function patchWebGPUForMaxLimits() {
  const gpu = navigator.gpu as GPU;
  const origRequestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (opts?: GPURequestAdapterOptions) => {
    const adapter = await origRequestAdapter(opts);
    if (!adapter) return adapter;
    const origRequestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = (desc: GPUDeviceDescriptor = {}) => {
      const requiredLimits: Record<string, number> = {};
      const proto = Object.getPrototypeOf(adapter.limits);
      for (const name of Object.getOwnPropertyNames(proto)) {
        const v = (adapter.limits as unknown as Record<string, unknown>)[name];
        if (typeof v === 'number') requiredLimits[name] = v;
      }
      return origRequestDevice({
        ...desc,
        requiredFeatures: [...adapter.features] as GPUFeatureName[],
        requiredLimits: { ...requiredLimits, ...(desc.requiredLimits ?? {}) },
      });
    };
    return adapter;
  };
}

function buildScene(renderer: THREE.WebGPURenderer) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e14);

  // Image-based lighting so metals actually reflect something (RoomEnvironment
  // ships with three — no extra dependency). Falls back to lights only if PMREM
  // fails in a stripped headless GPU.
  try {
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  } catch (e) {
    log(`environment fallback (${(e as Error).message})`);
  }
  scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x20303a, 0.6));
  const key = new THREE.DirectionalLight(0xfff2e0, 3.0);
  key.position.set(4, 6, 3);
  scene.add(key);
  const warm = new THREE.PointLight(0xffaa55, 40, 30);
  warm.position.set(-3, 2, 2);
  scene.add(warm);
  const cool = new THREE.PointLight(0x55aaff, 30, 30);
  cool.position.set(3, 1.5, -2);
  scene.add(cool);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardNodeMaterial({ color: 0x2a2f3a, roughness: 0.55, metalness: 0.1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.9;
  scene.add(ground);

  // A thin-featured torus knot is aliasing-hostile — good for showing FSR edge
  // quality — plus a few varied-material spheres for specular highlights (HDR
  // values >1 that the denoiser's hdr model + ACES tonemap actually exercise).
  const knot = new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.75, 0.24, 220, 32),
    new THREE.MeshStandardNodeMaterial({ color: 0xe0554f, roughness: 0.15, metalness: 0.9 }),
  );
  knot.position.y = 0.35;
  scene.add(knot);

  const specs = [
    { c: 0x38a169, r: 0.15, m: 0.0, x: -1.9, s: 0.6 },
    { c: 0xd6b24a, r: 0.1, m: 1.0, x: 1.9, s: 0.7 },
    { c: 0x8a6cff, r: 0.4, m: 0.2, x: 0.1, s: 0.45, z: 1.7 },
  ];
  for (const sp of specs) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(sp.s, 48, 48),
      new THREE.MeshStandardNodeMaterial({ color: sp.c, roughness: sp.r, metalness: sp.m }),
    );
    m.position.set(sp.x, sp.s - 0.9, sp.z ?? 0);
    scene.add(m);
  }

  const camera = new THREE.PerspectiveCamera(50, DW / DH, 0.1, 100);
  camera.position.set(0, 1.2, 4.2);
  camera.lookAt(0, 0.2, 0);
  return { scene, camera, knot };
}

async function main() {
  if (!(await ensureWebGPU())) return;
  patchWebGPUForMaxLimits();

  // 1) Denoiser FIRST — ORT creates the GPUDevice the whole pipeline shares.
  //    Color-only HDR path (no aux) keeps the demo dependency-light and robust;
  //    the pipeline story is the three-package chain + per-stage cost, not aux.
  const denoiser = await Denoiser.create({ quality: 'balanced' });
  const device = denoiser.device;
  log('denoiser ready — GPUDevice created; three.js + upscaler will share it');
  device.lost.then((info) => log(`DEVICE LOST: ${info.reason} — ${info.message}`, true));

  // 2) three.js WebGPURenderer on the SAME device.
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
  const renderer = new THREE.WebGPURenderer({ canvas, device, antialias: false });
  await renderer.init();
  renderer.setSize(DW, DH, false);
  // The denoiser + upscaler own all colour management (ACES tonemap + sRGB); every
  // quad we draw uses `fragmentNode` (a raw write, no material transforms), so keep
  // the renderer's own output transform identity.
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const backendGet = (o: unknown): GPUTexture | undefined =>
    (renderer.backend as unknown as { get: (o: unknown) => { texture?: GPUTexture } }).get(o)?.texture;

  const { scene, camera, knot } = buildScene(renderer);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.2, 0);
  controls.update();

  // 3) @pmndrs/upscaler on the SAME device (it grabs renderer.backend.device).
  const upscaler = new Upscaler({ renderer });
  upscaler.init();
  // Explicit render size: our input is produced by an external pass (denoiser)
  // whose resolution we control, so pin render + display exactly. Spatial path =
  // FSR1 (EASU + RCAS), no motion vectors.
  upscaler.configure({ displayWidth: DW, displayHeight: DH, renderWidth: RW, renderHeight: RH, path: 'spatial' });
  // The input is denoised but can carry faint residual grain; RCAS's denoise
  // variant stops the sharpener amplifying lone luma outliers (its docs call out
  // exactly this "pairs with a spatial denoiser upstream" case).
  upscaler.settings.rcasDenoise = true;

  // ---- render targets --------------------------------------------------------
  // linear HDR scene render → linear HDR noisy → (denoise) display-encoded result.
  const cleanRT = new THREE.RenderTarget(RW, RH, { type: THREE.HalfFloatType });
  const noisyRT = new THREE.RenderTarget(RW, RH, { type: THREE.HalfFloatType, depthBuffer: false });
  const displayRT = new THREE.RenderTarget(RW, RH, { type: THREE.HalfFloatType, depthBuffer: false });
  // Denoiser output target (caller-owned, STORAGE_BINDING). rgba16float holds the
  // ACES+sRGB display-encoded result the FSR1 spatial path expects (FLAG_INPUT_DISPLAY).
  const denoisedTex = new THREE.StorageTexture(RW, RH);
  denoisedTex.type = THREE.HalfFloatType;
  denoisedTex.colorSpace = THREE.NoColorSpace;
  denoisedTex.generateMipmaps = false;
  // Present render-res sources with nearest filtering so "upscale off" is an honest
  // blocky nearest-neighbour comparison against FSR (the upscaler uses its own
  // internal sampler, so this never affects the FSR path).
  for (const t of [denoisedTex, displayRT.texture]) {
    t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestFilter;
  }
  renderer.initTexture(denoisedTex);
  const denoisedGpuTex = backendGet(denoisedTex)!;

  // ---- TSL passes ------------------------------------------------------------
  const uSpp = uniform(6);
  const uFrame = uniform(0);
  // Synthetic path-tracer-style noise: relative (luminance-scaled) grain whose
  // amplitude falls as 1/√spp — the same "more samples → less noise" curve real
  // Monte-Carlo rendering follows, giving the denoiser genuine work.
  const noiseFrag = Fn(() => {
    const base = texture(cleanRT.texture, uv());
    const px = screenCoordinate;
    const cell = px.x.floor().add(px.y.floor().mul(RW)).add(uFrame.mul(RW * RH + 1));
    const sigma = float(0.55).div(uSpp.max(1).sqrt());
    const n = vec3(hash(cell.add(0.5)), hash(cell.add(1.5)), hash(cell.add(2.5))).mul(2).sub(1);
    return vec4(base.rgb.mul(sigma.mul(n).add(1)).max(0), 1);
  });
  // ACES (Narkowicz) tonemap + sRGB encode — matches the denoiser's 'aces-srgb'
  // transfer, so the denoise-on and denoise-off images are colour-consistent.
  const acesSrgb = Fn(([lin]: [ReturnType<typeof vec3>]) => {
    const v = lin.max(0);
    const tm = v.mul(v.mul(2.51).add(0.03)).div(v.mul(v.mul(2.43).add(0.59)).add(0.14)).clamp(0, 1);
    const lo = tm.mul(12.92);
    const hi = tm.pow(1 / 2.4).mul(1.055).sub(0.055);
    return mix(lo, hi, step(0.0031308, tm));
  });
  const displayFrag = Fn(() => vec4(acesSrgb(texture(noisyRT.texture, uv()).rgb), 1));

  const noiseQuad = new THREE.QuadMesh(Object.assign(new THREE.NodeMaterial(), { fragmentNode: noiseFrag() }));
  const displayQuad = new THREE.QuadMesh(Object.assign(new THREE.NodeMaterial(), { fragmentNode: displayFrag() }));
  // Present quad: raw texture write to the canvas. Swap the source per frame.
  const presentTexNode = texture(denoisedTex);
  const presentQuad = new THREE.QuadMesh(Object.assign(new THREE.NodeMaterial(), { fragmentNode: presentTexNode }));

  // ---- controls --------------------------------------------------------------
  let denoiseOn = true, upscaleOn = true;
  const wireSeg = (id: string, on: () => void, off: () => void) => {
    const seg = document.querySelector<HTMLDivElement>(id)!;
    seg.querySelectorAll<HTMLButtonElement>('button').forEach((b) => {
      b.addEventListener('click', () => {
        seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
        b.setAttribute('aria-pressed', 'true');
        (b.dataset.v === 'on' ? on : off)();
      });
    });
  };
  wireSeg('#denoise-seg', () => (denoiseOn = true), () => (denoiseOn = false));
  wireSeg('#upscale-seg', () => (upscaleOn = true), () => (upscaleOn = false));
  const sppEl = document.querySelector<HTMLInputElement>('#spp')!;
  sppEl.addEventListener('input', () => {
    uSpp.value = parseInt(sppEl.value, 10);
    document.querySelector('#spp-val')!.textContent = sppEl.value;
  });
  const sharpEl = document.querySelector<HTMLInputElement>('#sharp')!;
  sharpEl.addEventListener('input', () => {
    upscaler.settings.sharpness = parseFloat(sharpEl.value);
    document.querySelector('#sharp-val')!.textContent = parseFloat(sharpEl.value).toFixed(2);
  });

  document.querySelector('#resline')!.innerHTML =
    `render <b>${RW}×${RH}</b> → display <b>${DW}×${DH}</b> · FSR1 spatial · <b>${(DW / RW).toFixed(1)}×</b> upscale`;

  // ---- per-stage cost UI -----------------------------------------------------
  const bars = {
    render: document.querySelector<HTMLElement>('#bar-render')!,
    denoise: document.querySelector<HTMLElement>('#bar-denoise')!,
    upscale: document.querySelector<HTMLElement>('#bar-upscale')!,
  };
  const msEls = {
    render: document.querySelector<HTMLElement>('#ms-render')!,
    denoise: document.querySelector<HTMLElement>('#ms-denoise')!,
    upscale: document.querySelector<HTMLElement>('#ms-upscale')!,
  };
  const totalEl = document.querySelector<HTMLElement>('#ms-total')!;
  const fpsEl = document.querySelector<HTMLElement>('#fps')!;
  const sm = { render: 0, denoise: 0, upscale: 0 };
  function paint(r: number, d: number, u: number) {
    sm.render = sm.render ? sm.render * 0.85 + r * 0.15 : r;
    sm.denoise = sm.denoise ? sm.denoise * 0.85 + d * 0.15 : d;
    sm.upscale = sm.upscale ? sm.upscale * 0.85 + u * 0.15 : u;
    const total = sm.render + sm.denoise + sm.upscale;
    const scale = Math.max(total, 1);
    bars.render.style.width = `${(sm.render / scale) * 100}%`;
    bars.denoise.style.width = `${(sm.denoise / scale) * 100}%`;
    bars.upscale.style.width = `${(sm.upscale / scale) * 100}%`;
    msEls.render.textContent = `${sm.render.toFixed(1)} ms`;
    msEls.denoise.textContent = denoiseOn ? `${sm.denoise.toFixed(1)} ms` : 'off';
    msEls.upscale.textContent = upscaleOn ? `${sm.upscale.toFixed(1)} ms` : 'off';
    totalEl.textContent = `${total.toFixed(1)} ms`;
    fpsEl.textContent = `${(1000 / Math.max(total, 0.001)).toFixed(0)} fps`;
  }

  // ---- the pipeline ----------------------------------------------------------
  // Each stage submits its GPU work then awaits completion, so the reported ms are
  // real per-stage GPU times, not just CPU encode time — the cost breakdown IS the
  // story here.
  let frame = 0;
  let inFlight = false;
  async function step_() {
    if (inFlight) return;
    inFlight = true;
    try {
      uFrame.value = frame++ % 64;

      // Stage 1 — render clean scene, then add noise (both linear HDR).
      const t0 = performance.now();
      renderer.setRenderTarget(cleanRT);
      renderer.render(scene, camera);
      renderer.setRenderTarget(noisyRT);
      noiseQuad.render(renderer);
      renderer.setRenderTarget(null);
      await device.queue.onSubmittedWorkDone();
      const renderMs = performance.now() - t0;

      // Stage 2 — denoise (→ display-encoded render-res) OR tonemap the noisy frame.
      let src: THREE.Texture;
      let denoiseMs = 0;
      if (denoiseOn) {
        const t1 = performance.now();
        await denoiser.denoiseTextures({
          color: backendGet(noisyRT.texture)!,
          output: denoisedGpuTex,
          hdr: true,
          inputFlipY: FLIP_INPUT,
          transfer: 'aces-srgb',
        });
        denoiseMs = performance.now() - t1;
        src = denoisedTex;
      } else {
        renderer.setRenderTarget(displayRT);
        displayQuad.render(renderer);
        renderer.setRenderTarget(null);
        await device.queue.onSubmittedWorkDone();
        src = displayRT.texture;
      }

      // Stage 3 — FSR1 spatial upscale to display res (or present render-res nearest).
      let upscaleMs = 0;
      let presentTex: THREE.Texture;
      if (upscaleOn) {
        const t2 = performance.now();
        upscaler.dispatch({ color: src }, camera);
        await device.queue.onSubmittedWorkDone();
        upscaleMs = performance.now() - t2;
        presentTex = upscaler.outputTexture;
      } else {
        presentTex = src;
      }

      // Present.
      presentTexNode.value = presentTex;
      presentQuad.render(renderer);

      paint(renderMs, denoiseMs, upscaleMs);
    } catch (e) {
      log('pipeline ERROR: ' + (e as Error).message, true);
      throw e;
    } finally {
      inFlight = false;
    }
  }

  // gentle idle spin so reflections/thin features move (shows FSR under motion)
  let spinning = true;
  let paused = false; // measure() pauses the rAF loop so it can drive frames itself
  controls.addEventListener('start', () => (spinning = false));
  function loop() {
    if (paused) { requestAnimationFrame(loop); return; }
    if (spinning) knot.rotation.y += 0.004;
    controls.update();
    step_().then(() => requestAnimationFrame(loop)).catch(() => {});
  }
  // Prime the backend textures once (so backend.get resolves) then start.
  renderer.setRenderTarget(cleanRT); renderer.render(scene, camera); renderer.setRenderTarget(null);
  log('pipeline ready — three.js → denoiser → @pmndrs/upscaler, one shared GPUDevice');
  loop();

  // ---- headless verification hooks ------------------------------------------
  const probeRT = new THREE.RenderTarget(DW, DH); // rgba8 for JS-side readback
  const probeTexNode = texture(denoisedTex);
  const probeQuad = new THREE.QuadMesh(Object.assign(new THREE.NodeMaterial(), { fragmentNode: probeTexNode }));
  async function readRGBA8(tex: THREE.Texture, w: number, h: number): Promise<Uint8ClampedArray> {
    probeTexNode.value = tex;
    renderer.setRenderTarget(probeRT);
    probeQuad.render(renderer);
    renderer.setRenderTarget(null);
    const px = await renderer.readRenderTargetPixelsAsync(probeRT, 0, 0, w, h);
    return px as unknown as Uint8ClampedArray;
  }
  const luma = (a: ArrayLike<number>, i: number) => 0.299 * a[i] + 0.587 * a[i + 1] + 0.114 * a[i + 2];
  // Mean 3×3 local luma variance over rows [y0,y1). Whole-frame = mixes real detail
  // with noise; a flat background strip isolates pure-noise cleaning.
  function localVar(px: ArrayLike<number>, w: number, y0: number, y1: number): number {
    let acc = 0, n = 0;
    for (let y = Math.max(1, y0); y < y1 - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        let s = 0, s2 = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const l = luma(px, ((y + dy) * w + (x + dx)) * 4); s += l; s2 += l * l;
        }
        const m = s / 9; acc += s2 / 9 - m * m; n++;
      }
    return acc / n;
  }
  // Mean |Δluma| across even 2×2-block column pairs (2i, 2i+1). A nearest 2× upscale
  // duplicates columns exactly, so within-block pairs are identical → ≈0. FSR
  // reconstructs distinct sub-pixel values → >0. Robust vs image smoothness (a flat
  // region is 0 for BOTH; the gap only opens where there is detail to resolve), so it
  // isolates "FSR produced genuine higher-res detail" from "nearest replicated blocks".
  function withinBlockDiff(px: ArrayLike<number>, w: number, h: number): number {
    let acc = 0, n = 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x + 1 < w; x += 2) {
        const i = (y * w + x) * 4;
        acc += Math.abs(luma(px, i) - luma(px, i + 4)); n++;
      }
    return acc / n;
  }
  function nonBlack(px: ArrayLike<number>, w: number, h: number): number {
    let nb = 0; const n = w * h;
    for (let i = 0; i < n; i++) if (luma(px, i * 4) > 6) nb++;
    return nb / n;
  }

  (window as unknown as Record<string, unknown>).__pipeline = {
    async measure() {
      // Pause the rAF loop and wait for any in-flight frame to drain, so our own
      // step_() calls actually execute (the inFlight guard would skip them otherwise).
      paused = true;
      for (let k = 0; k < 200 && inFlight; k++) await new Promise((r) => setTimeout(r, 25));
      // settle a few frames per config (first denoise also compiles the ORT graph)
      const settle = async () => { for (let k = 0; k < 4; k++) { await step_(); } };
      const BG = Math.floor(RH * 0.16); // top strip = flat background (no scene detail)

      // denoise ON, upscale ON. Read the SAME denoised source three ways so every
      // comparison is like-for-like (no confounding of denoise with upscale).
      denoiseOn = true; upscaleOn = true; await settle();
      const fsr = await readRGBA8(upscaler.outputTexture, DW, DH);   // FSR of denoised
      const nearestDenoised = await readRGBA8(denoisedTex, DW, DH);  // nearest 2× of the SAME denoised source
      const denoisedRR = await readRGBA8(denoisedTex, RW, RH);       // denoised render-res

      // denoise OFF at the same spp → the noisy render-res frame, for the noise delta.
      denoiseOn = false; upscaleOn = false; await settle();
      const noisyRR = await readRGBA8(displayRT.texture, RW, RH);

      // restore defaults
      denoiseOn = true; upscaleOn = true; paused = false;
      const out = {
        render: { w: RW, h: RH, nonBlack: +nonBlack(denoisedRR, RW, RH).toFixed(3) },
        denoise: {
          noisyVarWhole: +localVar(noisyRR, RW, 0, RH).toFixed(1),
          denoisedVarWhole: +localVar(denoisedRR, RW, 0, RH).toFixed(1),
          noisyVarBg: +localVar(noisyRR, RW, 0, BG).toFixed(1),
          denoisedVarBg: +localVar(denoisedRR, RW, 0, BG).toFixed(1),
        },
        upscale: {
          outW: DW, outH: DH,
          fsrNonBlack: +nonBlack(fsr, DW, DH).toFixed(3),
          // same denoised source: nearest replicates 2×2 blocks (≈0), FSR resolves detail (>0)
          fsrWithinBlockDiff: +withinBlockDiff(fsr, DW, DH).toFixed(3),
          nearestWithinBlockDiff: +withinBlockDiff(nearestDenoised, DW, DH).toFixed(3),
        },
        timings: { render: +sm.render.toFixed(1), denoise: +sm.denoise.toFixed(1), upscale: +sm.upscale.toFixed(1) },
      };
      (window as unknown as Record<string, unknown>).__measurement = out;
      log(`measured: bg noise var ${out.denoise.noisyVarBg}→${out.denoise.denoisedVarBg}, ` +
        `FSR detail ${out.upscale.fsrWithinBlockDiff} vs nearest ${out.upscale.nearestWithinBlockDiff}`);
      return out;
    },
  };
  (window as unknown as Record<string, unknown>).__ready = true;
}

// three's raster render targets read top-down for the denoiser (no vertical flip),
// same convention the pathtracer example's rasterized G-buffer uses.
const FLIP_INPUT = false;

demoFooter('upscale-pipeline');
main().catch((e) => log('ERROR: ' + (e as Error).message, true));
