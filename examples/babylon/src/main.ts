// Babylon.js integration demo — the honest CPU-copy path.
//
// Babylon renders a PBR showcase on ITS OWN WebGPU device; the denoiser runs on a
// SEPARATE ORT-owned device. Neither engine can adopt the other's GPUDevice
// (spike verdict in docs/status/phase-b-plan.md B2#5), so every frame crosses the
// boundary twice on the CPU:
//
//   Babylon → RenderTargetTexture.readPixels()  (GPU → CPU)   [readback ms]
//           → denoiser.denoise(bytes)            (CPU → ORT device → CPU) [denoise ms]
//           → RawTexture.update(result)          (CPU → GPU)   [upload ms]
//
// The two copies ARE the story — this file times them separately. Left panel is a
// 2D canvas of the exact bytes fed to the denoiser; the Babylon <canvas> on the
// right shows the denoised result uploaded back to Babylon's device (a fullscreen
// Layer over a RawTexture) — proof the round-trip closed on the GPU.

import {
  WebGPUEngine,
  Engine,
  type AbstractEngine,
  Scene,
  ArcRotateCamera,
  Camera,
  Vector3,
  Color3,
  Color4,
  HemisphericLight,
  DirectionalLight,
  MeshBuilder,
  PBRMaterial,
  ShadowGenerator,
  RenderTargetTexture,
  RawTexture,
  RawCubeTexture,
  Constants,
  ImageProcessingConfiguration,
  Texture,
  SSAO2RenderingPipeline,
  type Mesh,
} from '@babylonjs/core';
import { Denoiser } from 'denoiser';
import { ensureWebGPU, demoFooter } from '../../_shared/chrome';

// ---- constants --------------------------------------------------------------

const SIZE = 512; // square; width % 64 === 0 hits Babylon's fast readback path.
const MESH_MASK = 0x1; // scene geometry — rendered by the offscreen readback camera

// ---- DOM --------------------------------------------------------------------

const noisyCanvas = document.querySelector<HTMLCanvasElement>('#canvas-noisy')!;
const denoisedCanvas = document.querySelector<HTMLCanvasElement>('#canvas-denoised')!;
const engineCanvas = document.querySelector<HTMLCanvasElement>('#engine-canvas')!;
const denoiseBtn = document.querySelector<HTMLButtonElement>('#denoise-btn')!;
const autoToggle = document.querySelector<HTMLInputElement>('#auto-toggle')!;
const noiseSlider = document.querySelector<HTMLInputElement>('#noise-slider')!;
const noiseLabel = document.querySelector<HTMLElement>('#noise-label')!;
const statusEl = document.querySelector<HTMLElement>('#status')!;
const engineTagEl = document.querySelector<HTMLElement>('#engine-tag')!;
const mReadback = document.querySelector<HTMLElement>('#m-readback')!;
const mDenoise = document.querySelector<HTMLElement>('#m-denoise')!;
const mUpload = document.querySelector<HTMLElement>('#m-upload')!;
const mTotal = document.querySelector<HTMLElement>('#m-total')!;
const mNoise = document.querySelector<HTMLElement>('#m-noise')!;

const noisyCtx = noisyCanvas.getContext('2d')!;
const denoisedCtx = denoisedCanvas.getContext('2d')!;
noisyCanvas.width = SIZE;
noisyCanvas.height = SIZE;
denoisedCanvas.width = SIZE;
denoisedCanvas.height = SIZE;

function setStatus(html: string): void {
  statusEl.innerHTML = html;
}

// ---- procedural HDR environment (self-contained; no asset fetch) ------------
// A soft sky gradient baked into a float cube so the PBR metals have something to
// reflect. Warm zenith, bright horizon, dim floor — HDR (>1) so highlights read.

function buildEnvironment(scene: Scene): RawCubeTexture {
  const s = 64;
  const zenith = [1.15, 1.05, 0.9];
  const horizon = [0.85, 0.9, 1.0];
  const floor = [0.05, 0.06, 0.08];
  const dirForFace = (face: number, u: number, v: number): Vector3 => {
    // u,v in [-1,1]; standard cube-face basis (+X,-X,+Y,-Y,+Z,-Z).
    switch (face) {
      case 0: return new Vector3(1, -v, -u);
      case 1: return new Vector3(-1, -v, u);
      case 2: return new Vector3(u, 1, v);
      case 3: return new Vector3(u, -1, -v);
      case 4: return new Vector3(u, -v, 1);
      default: return new Vector3(-u, -v, -1);
    }
  };
  const faces: Float32Array[] = [];
  for (let face = 0; face < 6; face++) {
    const data = new Float32Array(s * s * 4);
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const u = (x + 0.5) / s * 2 - 1;
        const v = (y + 0.5) / s * 2 - 1;
        const d = dirForFace(face, u, v);
        d.normalize();
        const t = Math.max(0, d.y); // up-ness
        const b = Math.max(0, -d.y); // down-ness
        const g = 1 - t - b; // horizon band
        const r = zenith[0] * t + horizon[0] * g + floor[0] * b;
        const gg = zenith[1] * t + horizon[1] * g + floor[1] * b;
        const bb = zenith[2] * t + horizon[2] * g + floor[2] * b;
        const i = (y * s + x) * 4;
        // a subtle warm "sun" lobe near +Y/+Z to give the specular a hotspot
        const sun = Math.pow(Math.max(0, d.y * 0.5 + d.z * 0.5 + d.x * 0.2), 40) * 6;
        data[i] = r + sun;
        data[i + 1] = gg + sun * 0.95;
        data[i + 2] = bb + sun * 0.8;
        data[i + 3] = 1;
      }
    }
    faces.push(data);
  }
  const cube = new RawCubeTexture(
    scene, faces, s,
    Constants.TEXTUREFORMAT_RGBA, Constants.TEXTURETYPE_FLOAT,
    true, false, Texture.TRILINEAR_SAMPLINGMODE,
  );
  cube.gammaSpace = false; // linear HDR
  return cube;
}

// ---- scene ------------------------------------------------------------------

interface SceneKit {
  scene: Scene;
  rtt: RenderTargetTexture;
  ssao: SSAO2RenderingPipeline;
  readCamera: Camera;
}

function buildScene(engine: AbstractEngine): SceneKit {
  const scene = new Scene(engine);
  scene.clearColor = new Color4(0.02, 0.025, 0.035, 1);

  // ACES tonemap + sRGB out, so the RGBA8 readback is display-ready sRGB bytes —
  // exactly what the denoiser's LDR image path expects (srgb: true, the default).
  const ip = scene.imageProcessingConfiguration;
  ip.toneMappingEnabled = true;
  ip.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES;
  ip.exposure = 1.15;
  ip.contrast = 1.1;

  scene.environmentTexture = buildEnvironment(scene);
  scene.environmentIntensity = 0.55;

  // Readback camera: renders the geometry (+SSAO) into the offscreen RTT. It's the
  // only camera — the engine's own <canvas> stays offscreen; both visible panels
  // are painted from the CPU side (2D canvas), which is the honest story here.
  const readCamera = new ArcRotateCamera('read', Math.PI / 2.3, Math.PI / 2.6, 6.2, new Vector3(0, 0.6, 0), scene);
  readCamera.fov = 0.7;
  readCamera.layerMask = MESH_MASK;
  readCamera.minZ = 0.1;
  readCamera.maxZ = 100;
  scene.activeCameras = [readCamera];

  // Lights — one key with soft contact-hardening shadows (a genuine spatial-noise
  // source at low filtering quality), plus a dim hemispheric fill.
  const hemi = new HemisphericLight('hemi', new Vector3(0.2, 1, 0.1), scene);
  hemi.intensity = 0.25;
  hemi.diffuse = new Color3(0.9, 0.95, 1);

  const key = new DirectionalLight('key', new Vector3(-0.6, -1, 0.5), scene);
  key.position = new Vector3(6, 12, -5);
  key.intensity = 3.2;
  key.diffuse = new Color3(1, 0.96, 0.9);

  const shadow = new ShadowGenerator(1024, key);
  shadow.useContactHardeningShadow = true;
  shadow.contactHardeningLightSizeUVRatio = 0.35; // large penumbra
  shadow.filteringQuality = ShadowGenerator.QUALITY_LOW; // few taps -> noisy penumbra
  shadow.bias = 0.008;

  // Ground.
  const ground = MeshBuilder.CreateGround('ground', { width: 24, height: 24 }, scene);
  ground.layerMask = MESH_MASK;
  ground.receiveShadows = true;
  const groundMat = new PBRMaterial('groundMat', scene);
  groundMat.albedoColor = new Color3(0.35, 0.36, 0.4);
  groundMat.metallic = 0.05;
  groundMat.roughness = 0.75;
  ground.material = groundMat;

  // A row of PBR spheres sweeping metallic/roughness, plus a torus knot hero.
  const meshes: Mesh[] = [ground];
  const palette = [
    new Color3(0.9, 0.3, 0.25), new Color3(0.95, 0.75, 0.2),
    new Color3(0.3, 0.7, 0.4), new Color3(0.25, 0.5, 0.9),
  ];
  for (let i = 0; i < 4; i++) {
    const sphere = MeshBuilder.CreateSphere(`s${i}`, { diameter: 1.3, segments: 32 }, scene);
    sphere.position = new Vector3((i - 1.5) * 1.75, 0.65, 1.4);
    sphere.layerMask = MESH_MASK;
    const mat = new PBRMaterial(`sMat${i}`, scene);
    mat.albedoColor = palette[i];
    mat.metallic = i % 2 === 0 ? 0.9 : 0.1;
    mat.roughness = 0.12 + i * 0.22;
    sphere.material = mat;
    shadow.addShadowCaster(sphere);
    meshes.push(sphere);
  }

  const knot = MeshBuilder.CreateTorusKnot('knot', { radius: 0.7, tube: 0.24, radialSegments: 128, tubularSegments: 32 }, scene);
  knot.position = new Vector3(0, 1.3, -0.8);
  knot.layerMask = MESH_MASK;
  const knotMat = new PBRMaterial('knotMat', scene);
  knotMat.albedoColor = new Color3(0.8, 0.82, 0.85);
  knotMat.metallic = 1.0;
  knotMat.roughness = 0.18;
  knot.material = knotMat;
  shadow.addShadowCaster(knot);
  meshes.push(knot);

  // Offscreen render target the readback camera draws into (RGBA8 = byte readback).
  const rtt = new RenderTargetTexture('readTarget', SIZE, scene, {
    generateDepthBuffer: true,
    generateMipMaps: false,
    type: Constants.TEXTURETYPE_UNSIGNED_BYTE,
    format: Constants.TEXTUREFORMAT_RGBA,
  });
  rtt.renderList = meshes;
  rtt.activeCamera = readCamera;
  rtt.clearColor = scene.clearColor;
  scene.customRenderTargets.push(rtt);
  // Route the readback camera's on-screen output to this RTT (carries the SSAO
  // post-process chain, which a bare rtt.render() would skip).
  readCamera.outputRenderTarget = rtt;

  // SSAO — the primary spatial-noise source. Low sample count + blur bypassed =
  // raw, un-smoothed ambient-occlusion grain, i.e. honest Monte-Carlo-like noise.
  const ssao = new SSAO2RenderingPipeline('ssao', scene, { ssaoRatio: 1, blurRatio: 1 }, [readCamera]);
  ssao.radius = 1.4;
  ssao.totalStrength = 1.3;
  ssao.base = 0.1;
  ssao.samples = 3;
  ssao.bypassBlur = true;
  ssao.expensiveBlur = false;
  ssao.minZAspect = 0.2;
  ssao.maxZ = 60;

  return { scene, rtt, ssao, readCamera };
}

// ---- noise metric -----------------------------------------------------------
// Mean absolute luminance difference between 4-neighbours, normalised to [0,1].
// High-frequency energy: noisy input scores high, a smooth denoise scores low.

function noiseMetric(data: Uint8ClampedArray, w: number, h: number): number {
  const lum = (i: number) => 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  let sum = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const c = lum(i);
      sum += Math.abs(c - lum(i - 4)) + Math.abs(c - lum(i + 4));
      n += 2;
    }
  }
  return sum / n / 255;
}

// ---- main -------------------------------------------------------------------

async function main(): Promise<void> {
  if (!(await ensureWebGPU())) return;
  demoFooter('babylon');

  setStatus('booting Babylon…');

  // Babylon on its own WebGPU device if it initialises cleanly; WebGL2 otherwise.
  // Either way the denoiser owns a SEPARATE WebGPU device — that separation is the
  // whole point of the CPU-copy path.
  let engine: AbstractEngine;
  let engineName: string;
  try {
    const webgpu = new WebGPUEngine(engineCanvas, { antialias: true, stencil: false });
    await webgpu.initAsync();
    engine = webgpu;
    engineName = 'WebGPU';
  } catch (err) {
    console.warn('[babylon] WebGPU engine init failed, falling back to WebGL2:', err);
    engine = new Engine(engineCanvas, true, { preserveDrawingBuffer: false });
    engineName = 'WebGL2';
  }
  engineTagEl.textContent = `Babylon ${Engine.Version} · ${engineName}`;

  const { scene, rtt, ssao, readCamera } = buildScene(engine);

  // The denoised bytes are uploaded back onto Babylon's OWN device via
  // RawTexture.update — the real CPU→GPU copy a compositing integration pays, and
  // what the "Upload" timer measures. We mirror the result to a 2D canvas for the
  // on-page display (reads cleaner than a second WebGPU surface).
  const denoisedTex = RawTexture.CreateRGBATexture(
    new Uint8Array(SIZE * SIZE * 4), SIZE, SIZE, engine, false, false, Texture.NEAREST_SAMPLINGMODE,
  );

  setStatus('loading denoiser model…');
  const denoiser = await Denoiser.create({ quality: 'fast' });

  // ---- the round-trip ----
  let busy = false;
  async function roundTrip(): Promise<void> {
    if (busy) return;
    busy = true;
    try {
      // 1. readback (GPU → CPU)
      const t0 = performance.now();
      const pixels = (await rtt.readPixels(0, 0, undefined, true)) as Uint8Array | null;
      const t1 = performance.now();
      if (!pixels) { busy = false; return; }
      const bytes = new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, pixels.byteLength);

      // Show the exact denoiser input on the left 2D canvas.
      noisyCtx.putImageData(new ImageData(new Uint8ClampedArray(bytes), SIZE, SIZE), 0, 0);
      const inNoise = noiseMetric(bytes, SIZE, SIZE);

      // 2. denoise (CPU → separate ORT device → CPU)
      const t2 = performance.now();
      const out = await denoiser.denoise({ data: bytes, width: SIZE, height: SIZE });
      const t3 = performance.now();
      if (!out) { busy = false; return; }

      // 3. upload back to Babylon's device (CPU → GPU), then flush so the copy is
      // actually issued this frame (not just queued) — that's what we time.
      const t4 = performance.now();
      denoisedTex.update(new Uint8Array(out.data.buffer));
      engine.flushFramebuffer();
      const t5 = performance.now();

      // Mirror the uploaded frame to the on-page 2D canvas.
      denoisedCtx.putImageData(new ImageData(new Uint8ClampedArray(out.data), SIZE, SIZE), 0, 0);

      const outNoise = noiseMetric(out.data, SIZE, SIZE);

      const readback = t1 - t0;
      const denoise = t3 - t2;
      const upload = t5 - t4;
      mReadback.textContent = readback.toFixed(1);
      mDenoise.textContent = denoise.toFixed(1);
      mUpload.textContent = upload.toFixed(1);
      mTotal.textContent = (readback + denoise + upload).toFixed(1);
      const drop = inNoise > 0 ? (100 * (1 - outNoise / inNoise)) : 0;
      mNoise.textContent = `${inNoise.toFixed(4)} → ${outNoise.toFixed(4)}  (−${drop.toFixed(0)}%)`;

      const st = denoiser.stats;
      setStatus(
        `model <span class="model">${denoiser.modelName ?? '?'}</span> · ` +
        `${SIZE}×${SIZE} · ${st ? `${st.tiles} tile${st.tiles === 1 ? '' : 's'}` : ''} · ` +
        `noise −${drop.toFixed(0)}%`,
      );

      // expose for headless verification
      (window as unknown as Record<string, unknown>).__lastResult = {
        readback, denoise, upload, total: readback + denoise + upload,
        inNoise, outNoise, drop, model: denoiser.modelName,
      };
    } catch (err) {
      setStatus(`<span style="color:#ff7b72">ERROR: ${(err as Error).message}</span>`);
      console.error(err);
    } finally {
      busy = false;
    }
  }

  // ---- controls ----
  denoiseBtn.addEventListener('click', () => { void roundTrip(); });

  let auto = false;
  autoToggle.addEventListener('change', () => {
    auto = autoToggle.checked;
    denoiseBtn.disabled = auto;
  });

  function applyNoise(): void {
    const v = Number(noiseSlider.value); // 0..100
    ssao.totalStrength = 0.3 + (v / 100) * 2.2;
    ssao.samples = v > 66 ? 2 : v > 33 ? 3 : 4;
    noiseLabel.textContent = v < 20 ? 'subtle' : v < 60 ? 'moderate' : 'heavy';
  }
  noiseSlider.addEventListener('input', applyNoise);
  applyNoise();

  // Keep the readback RTT fresh every frame; drive round-trips from an auto loop
  // (chained, never overlapping) plus a slow camera drift so the scene is alive.
  let angle = readCamera instanceof ArcRotateCamera ? readCamera.alpha : 0;
  engine.runRenderLoop(() => {
    if (readCamera instanceof ArcRotateCamera) {
      angle += 0.0016;
      readCamera.alpha = angle;
    }
    scene.render();
    if (auto && !busy) void roundTrip();
  });

  // Prime the first result so both panels are populated on load.
  await roundTrip();
  autoToggle.disabled = false;
  denoiseBtn.disabled = false;
  (window as unknown as Record<string, unknown>).__ready = true;
}

main().catch((err) => {
  setStatus(`<span style="color:#ff7b72">ERROR: ${(err as Error).message}</span>`);
  console.error(err);
});
