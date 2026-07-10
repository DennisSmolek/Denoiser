// realtime-compare (Phase B2 demo #7): honest positioning of our OIDN full-frame
// denoiser vs three.js r185's temporal screen-space `recurrentDenoise()`.
//
// The fair fight (per docs/status/phase-b-plan.md #7): recurrentDenoise() is a
// SCREEN-SPACE-EFFECT denoiser, not a beauty-pass denoiser. So we build a noisy
// stochastic-SSR effect and denoise it BOTH ways on ONE shared GPUDevice:
//   LEFT  — SSRNode -> temporalReproject -> recurrentDenoise (mode:'specular'),
//           the canonical r185 pipeline (webgpu_postprocessing_ssr_denoise.html).
//           Temporal, runs every frame, tracks motion, needs history.
//   RIGHT — the SAME raw noisy SSR composited into the beauty frame, then the
//           WHOLE frame denoised by our OIDN network (`denoiseTextures`, hdr,
//           optional albedo+normal aux) on a cadence (every N frames).
//
// Device rule (onnxruntime #26107): the denoiser owns the GPUDevice; three.js
// borrows it. Denoiser.create() FIRST, then new WebGPURenderer({ device }).
import * as THREE from 'three/webgpu';
import {
  pass, mrt, output, diffuseColor, normalView, materialMetalness, materialRoughness,
  packNormalToRGB, unpackRGBToNormal, velocity, sample, texture, vec2, vec4, renderOutput,
} from 'three/tsl';
import { ssr } from 'three/addons/tsl/display/SSRNode.js';
import { temporalReproject } from 'three/addons/tsl/display/TemporalReprojectNode.js';
import { recurrentDenoise } from 'three/addons/tsl/display/RecurrentDenoiseNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Denoiser } from 'denoiser';
import { ensureWebGPU, demoFooter } from '../../_shared/chrome';

const RES = 512;
const statusEl = document.querySelector<HTMLPreElement>('#status')!;
const hudLeft = document.querySelector<HTMLDivElement>('#hud-left')!;
const hudRight = document.querySelector<HTMLDivElement>('#hud-right')!;
const log = (m: string) => { statusEl.textContent = m; console.log('[realtime-compare]', m); };

const params = new URLSearchParams(location.search);
const headless = params.has('headless'); // skip on-canvas WebGPU contexts (they stall headless)

// Patch requestDevice to request the adapter's MAX limits/features BEFORE any
// device is created — ORT (which makes the shared device) otherwise requests a
// minimal one and three's heavier pipelines fail validation. (onnxruntime #26107.)
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

// Procedural equirect environment (gradient sky + warm sun): IBL for the metals
// AND SSR's env fallback for rays that leave the screen — this is what makes the
// stochastic reflections carry visible, denoisable noise. (The canonical example
// loads an HDR here; we generate one so the demo needs no assets.)
function makeEnvironment(): THREE.DataTexture {
  const W = 512, H = 256;
  const data = new Float32Array(W * H * 4);
  const sunDir = new THREE.Vector3(0.4, 0.5, -0.2).normalize();
  for (let y = 0; y < H; y++) {
    const v = y / (H - 1);
    const phi = v * Math.PI; // 0=zenith .. PI=nadir
    for (let x = 0; x < W; x++) {
      const u = x / (W - 1);
      const theta = u * Math.PI * 2;
      const dir = new THREE.Vector3(Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta));
      // sky gradient
      const t = Math.max(0, dir.y);
      let r = 0.10 + 0.35 * t, g = 0.16 + 0.45 * t, b = 0.30 + 0.55 * t;
      // ground haze
      if (dir.y < 0) { const k = Math.min(1, -dir.y * 2); r = r * (1 - k) + 0.06 * k; g = g * (1 - k) + 0.05 * k; b = b * (1 - k) + 0.05 * k; }
      // warm sun disk + glow (bright -> HDR, drives noisy specular)
      const d = dir.dot(sunDir);
      const sun = Math.pow(Math.max(0, d), 800) * 40 + Math.pow(Math.max(0, d), 8) * 1.2;
      r += sun * 1.0; g += sun * 0.85; b += sun * 0.6;
      const i = (y * W + x) * 4;
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 1;
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function buildScene(): { scene: THREE.Scene; camera: THREE.PerspectiveCamera; envTex: THREE.DataTexture } {
  const scene = new THREE.Scene();
  const envTex = makeEnvironment();
  scene.environment = envTex;
  scene.background = envTex;

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(3.2, 1.9, 4.4);
  camera.lookAt(0, 0.6, 0);

  // Reflective floor: metallic + MODERATE roughness so the stochastic SSR GGX
  // lobe is wide -> reflections are visibly NOISY (the point of the demo).
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0x223040, metalness: 1.0, roughness: 0.38 }),
  );
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  // A back wall for off-screen rays to reflect into (more reflection coverage).
  const wall = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 20),
    new THREE.MeshStandardMaterial({ color: 0x14202e, metalness: 0.6, roughness: 0.5 }),
  );
  wall.position.set(0, 5, -6);
  scene.add(wall);

  const specs: Array<[number, number, number, number, number]> = [
    // color, x, roughness, metalness, y-radius
    [0xe5484d, -1.8, 0.25, 1.0, 0.7],
    [0x30a46c, 0.0, 0.45, 1.0, 0.8],
    [0x4493f8, 1.9, 0.15, 1.0, 0.6],
  ];
  for (const [c, x, rough, metal, r] of specs) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(r, 48, 48),
      new THREE.MeshStandardMaterial({ color: c, roughness: rough, metalness: metal }),
    );
    m.position.set(x, r, 0);
    scene.add(m);
  }
  const knot = new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.55, 0.18, 160, 24),
    new THREE.MeshStandardMaterial({ color: 0xf5a524, roughness: 0.35, metalness: 1.0 }),
  );
  knot.position.set(0.2, 1.9, -1.4);
  scene.add(knot);

  // Analytic lights (WebGPURenderer supports them) + a warm/cool fill so the
  // metals pick up colored highlights that then smear into the reflections.
  scene.add(new THREE.AmbientLight(0x334455, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 3.0);
  key.position.set(4, 6, 3);
  scene.add(key);
  const warm = new THREE.PointLight(0xffaa55, 12, 30);
  warm.position.set(-3, 2.5, 2);
  scene.add(warm);
  const cool = new THREE.PointLight(0x55aaff, 12, 30);
  cool.position.set(3, 3, -2);
  scene.add(cool);

  return { scene, camera, envTex };
}

async function main() {
  if (!(await ensureWebGPU())) return;
  patchWebGPUForMaxLimits();

  // 1) Denoiser first: ORT owns the GPUDevice we then share with three.js.
  const denoiser = await Denoiser.create({ precision: 'fp16' });
  const device = denoiser.device;
  device.lost.then((info) => log(`DEVICE LOST: ${info.reason} — ${info.message}`));
  log(`denoiser ready — sharing GPUDevice with three.js`);

  // 2) three.js WebGPURenderer on the SAME device. Offscreen canvas at RES; the
  // SSR/temporalReproject/recurrentDenoise nodes self-size to the drawing buffer.
  const glCanvas = document.createElement('canvas');
  const renderer = new THREE.WebGPURenderer({ canvas: glCanvas, antialias: false, device });
  await renderer.init();
  renderer.setSize(RES, RES, false);

  const { scene, camera, envTex } = buildScene();
  const controls = new OrbitControls(camera, headless ? glCanvas : document.querySelector<HTMLCanvasElement>('#canvas-left')!);
  controls.target.set(0, 0.6, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.6;
  controls.update();

  // 3) Canonical r185 SSR-denoise graph (webgpu_postprocessing_ssr_denoise.html).
  const scenePass = pass(scene, camera);
  scenePass.setMRT(mrt({
    output,
    // albedo.rgb + metalness.a
    diffuseColor: vec4(diffuseColor.rgb, materialMetalness),
    // packNormalToRGB(normal).rgb + roughness.a
    normal: vec4(packNormalToRGB(normalView).rgb, materialRoughness),
    velocity,
  }));
  const sceneColor = scenePass.getTextureNode('output');
  const sceneNormalTex = scenePass.getTextureNode('normal');
  const sceneDepth = scenePass.getTextureNode('depth');
  const sceneVelocity = scenePass.getTextureNode('velocity');
  const sceneDiffuse = scenePass.getTextureNode('diffuseColor');
  const sceneNormal = sample((uv) => unpackRGBToNormal(sceneNormalTex.sample(uv).rgb));
  const metalRough = sample((uv) => vec2(sceneDiffuse.sample(uv).a, sceneNormalTex.sample(uv).a));

  const ssrNode = ssr(sceneColor, sceneDepth, sceneNormal, {
    stochastic: true,
    diffuseNode: sceneDiffuse,
    metalnessNode: sceneDiffuse.a,
    roughnessNode: sceneNormalTex.a,
    envImportanceSampling: true,
  });
  ssrNode.setEnvMap(envTex); // env fallback for rays that leave the screen (noisy specular)
  const reproj = temporalReproject(ssrNode, sceneDepth, sceneNormalTex, sceneVelocity, camera, {
    mode: 'specular',
    accumulate: false,
  });
  const denoiseNode = recurrentDenoise(reproj, camera, {
    depth: sceneDepth,
    normal: sceneNormalTex,
    raw: ssrNode,
    metalRoughness: metalRough,
    mode: 'specular',
    accumulate: true,
  });
  denoiseNode.alphaSource = 'raylength';
  // Feedback: denoise output becomes SSR's + temporalReproject's history.
  ssrNode.setHistory(denoiseNode, sceneVelocity);
  reproj.setHistoryTexture(denoiseNode);

  // Underlying three.Textures (populated during the heavy pass) — sampled as PLAIN
  // texture nodes in the raw-composite pass so it does NOT re-trigger the scene /
  // SSR / denoise FRAME nodes (no double scene render, no extra SSR accumulation).
  const sceneOutTexture = scenePass.getTexture('output');
  const ssrRawTexture = (ssrNode as unknown as { _ssrRenderTarget: THREE.RenderTarget })._ssrRenderTarget.texture;

  // --- render targets ---
  const HALF = { type: THREE.HalfFloatType as THREE.TextureDataType, depthBuffer: false };
  const leftLinearRT = new THREE.RenderTarget(RES, RES, HALF);   // beauty + DENOISED reflections (linear HDR)
  const rawLinearRT = new THREE.RenderTarget(RES, RES, HALF);    // beauty + RAW reflections (linear HDR) -> OIDN input
  const leftDisplayRT = new THREE.RenderTarget(RES, RES, { depthBuffer: false }); // rgba8unorm display
  const rawDisplayRT = new THREE.RenderTarget(RES, RES, { depthBuffer: false });  // rgba8unorm display (right pre-denoise)
  const gbuffer = new THREE.RenderTarget(RES, RES, { count: 2, type: THREE.HalfFloatType, depthBuffer: true });
  gbuffer.textures[0].name = 'albedo';
  gbuffer.textures[1].name = 'normal';

  // --- quad passes ---
  const heavyMat = new THREE.NodeMaterial();
  // Reference ssrNode DIRECTLY (the canonical alpha gate) so its updateBefore is
  // collected and it actually traces — reaching it only through the passTexture
  // severance of denoise/reproj leaves SSR unrendered (black). Composite exactly
  // like webgpu_postprocessing_ssr_denoise.html: beauty + gated denoised reflections.
  const denoiseBlend = denoiseNode.rgb.mul(ssrNode.a.greaterThan(0));
  heavyMat.fragmentNode = vec4(sceneColor.rgb.add(denoiseBlend), 1);
  const heavyQuad = new THREE.QuadMesh(heavyMat);

  const rawMat = new THREE.NodeMaterial();
  rawMat.fragmentNode = vec4(texture(sceneOutTexture).rgb.add(texture(ssrRawTexture).rgb), 1);
  const rawQuad = new THREE.QuadMesh(rawMat);

  const tmLeftMat = new THREE.NodeMaterial();
  tmLeftMat.fragmentNode = renderOutput(texture(leftLinearRT.texture), THREE.ACESFilmicToneMapping, THREE.SRGBColorSpace);
  const tmLeftQuad = new THREE.QuadMesh(tmLeftMat);

  const tmRawMat = new THREE.NodeMaterial();
  tmRawMat.fragmentNode = renderOutput(texture(rawLinearRT.texture), THREE.ACESFilmicToneMapping, THREE.SRGBColorSpace);
  const tmRawQuad = new THREE.QuadMesh(tmRawMat);

  // DEBUG: SSR reflections only (no scene beauty) — diagnostic for the shot hook.
  const tmSsrMat = new THREE.NodeMaterial();
  tmSsrMat.fragmentNode = renderOutput(vec4(texture(ssrRawTexture).rgb.mul(4), 1), THREE.ACESFilmicToneMapping, THREE.SRGBColorSpace);
  const tmSsrQuad = new THREE.QuadMesh(tmSsrMat);
  const ssrDbgRT = new THREE.RenderTarget(RES, RES, { depthBuffer: false });

  const backendGet = (o: unknown): GPUTexture | undefined =>
    (renderer.backend as unknown as { get: (o: unknown) => { texture?: GPUTexture } }).get(o)?.texture;

  function renderToRT(rt: THREE.RenderTarget | null, quad: THREE.QuadMesh) {
    renderer.setRenderTarget(rt);
    quad.render(renderer);
    renderer.setRenderTarget(null);
  }
  function renderAuxGBuffer() {
    renderer.setMRT(mrt({ albedo: diffuseColor, normal: normalView }));
    renderer.setRenderTarget(gbuffer);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.setMRT(null);
  }

  // --- on-canvas blit (skipped in headless; those getContext('webgpu') calls stall) ---
  const leftCanvas = document.querySelector<HTMLCanvasElement>('#canvas-left')!;
  const rightCanvas = document.querySelector<HTMLCanvasElement>('#canvas-right')!;
  leftCanvas.width = leftCanvas.height = rightCanvas.width = rightCanvas.height = RES;
  let leftCtx: GPUCanvasContext | null = null;
  let rightCtx: GPUCanvasContext | null = null;
  if (!headless) {
    for (const [cv, set] of [[leftCanvas, (c: GPUCanvasContext) => (leftCtx = c)], [rightCanvas, (c: GPUCanvasContext) => (rightCtx = c)]] as const) {
      const ctx = cv.getContext('webgpu');
      if (ctx) { ctx.configure({ device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }); set(ctx); }
    }
  }
  function blit(src: GPUTexture, ctx: GPUCanvasContext | null) {
    if (!ctx) return;
    const enc = device.createCommandEncoder();
    enc.copyTextureToTexture({ texture: src }, { texture: ctx.getCurrentTexture() }, { width: RES, height: RES });
    device.queue.submit([enc.finish()]);
  }

  // --- controls / UI ---
  const motionBox = document.querySelector<HTMLInputElement>('#motion')!;
  const auxBox = document.querySelector<HTMLInputElement>('#aux')!;
  const cadenceInput = document.querySelector<HTMLInputElement>('#cadence')!;
  const cadenceVal = document.querySelector<HTMLSpanElement>('#cadence-val')!;
  const refBtn = document.querySelector<HTMLButtonElement>('#ref-btn')!;
  const cadence = () => Math.max(1, parseInt(cadenceInput.value, 10) || 6);
  cadenceInput.addEventListener('input', () => { cadenceVal.textContent = `every ${cadence()} frames`; });
  motionBox.addEventListener('change', () => { controls.autoRotate = motionBox.checked; });
  controls.autoRotate = motionBox.checked;

  // --- our denoise (cadence) ---
  let ourBusy = false;
  let ourMs = 0;
  let hasDenoised = false;
  let lastOut: GPUTexture | undefined;
  async function runOurDenoise(): Promise<GPUTexture | undefined> {
    const color = backendGet(rawLinearRT.texture);
    if (!color) throw new Error('raw color texture unavailable');
    let albedo: GPUTexture | undefined;
    let normal: GPUTexture | undefined;
    if (auxBox.checked) {
      renderAuxGBuffer();
      albedo = backendGet(gbuffer.textures[0]);
      normal = backendGet(gbuffer.textures[1]);
    }
    const t0 = performance.now();
    const outTex = await denoiser.denoiseTextures({
      color, albedo, normal,
      hdr: true,
      inputFlipY: false,      // rasterized quad target -> top-down
      auxInputFlipY: false,   // rasterized G-buffer   -> top-down
      transfer: 'aces-srgb',  // display-ready bytes
    });
    ourMs = performance.now() - t0;
    if (outTex) { hasDenoised = true; lastOut = outTex; if (!headless) blit(outTex, rightCtx); }
    return outTex;
  }

  // --- approximate PSNR vs a cheap converged reference ---
  // Accumulate the raw composite over K static frames (running mean, on the GPU is
  // overkill for a demo — we average rgba8 tonemapped reads on the CPU), then PSNR
  // of each side's display bytes vs that reference. Best-effort; labeled approximate.
  async function readRT(rt: THREE.RenderTarget): Promise<Uint8ClampedArray> {
    const tex = backendGet(rt.texture)!;
    const rowBytes = RES * 4;
    const buf = device.createBuffer({ size: rowBytes * RES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: rowBytes }, { width: RES, height: RES });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8ClampedArray(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    return bytes;
  }
  async function readTex(tex: GPUTexture): Promise<Uint8ClampedArray> {
    const rowBytes = RES * 4;
    const buf = device.createBuffer({ size: rowBytes * RES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: rowBytes }, { width: RES, height: RES });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8ClampedArray(buf.getMappedRange().slice(0));
    buf.unmap(); buf.destroy();
    return bytes;
  }
  const luma = (b: Uint8ClampedArray, i: number) => 0.299 * b[i] + 0.587 * b[i + 1] + 0.114 * b[i + 2];
  function psnr(a: Uint8ClampedArray, ref: Float32Array): number {
    let mse = 0;
    for (let i = 0; i < RES * RES; i++) {
      for (let c = 0; c < 3; c++) { const d = a[i * 4 + c] - ref[i * 3 + c]; mse += d * d; }
    }
    mse /= RES * RES * 3;
    return mse < 1e-6 ? 99 : 10 * Math.log10((255 * 255) / mse);
  }
  // mean 3x3 local luma variance (speckle -> high)
  function localVar(b: Uint8ClampedArray): number {
    let acc = 0, n = 0;
    for (let y = 1; y < RES - 1; y++) for (let x = 1; x < RES - 1; x++) {
      let s = 0, s2 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const l = luma(b, ((y + dy) * RES + (x + dx)) * 4); s += l; s2 += l * l;
      }
      const m = s / 9; acc += s2 / 9 - m * m; n++;
    }
    return acc / n;
  }
  const nonBlack = (b: Uint8ClampedArray): number => {
    let n = 0; for (let i = 0; i < RES * RES; i++) if (b[i * 4] + b[i * 4 + 1] + b[i * 4 + 2] > 12) n++;
    return n / (RES * RES);
  };

  let psnrLeft = NaN, psnrRight = NaN;
  async function measurePSNR(frames = 96): Promise<{ left: number; right: number }> {
    const wasMotion = controls.autoRotate;
    controls.autoRotate = false;
    refBtn.disabled = true; refBtn.textContent = 'measuring…';
    const ref = new Float32Array(RES * RES * 3);
    // Let temporal history settle, then average the tonemapped raw composite.
    for (let f = 0; f < frames; f++) {
      await renderFrame();
      renderToRT(rawDisplayRT, tmRawQuad);
      const b = await readRT(rawDisplayRT);
      const w = f < frames / 3 ? 0 : 1; // discard the first third (warm-up)
      if (w) for (let i = 0; i < RES * RES * 3; i++) ref[i] += b[Math.floor(i / 3) * 4 + (i % 3)];
    }
    const kept = frames - Math.floor(frames / 3);
    for (let i = 0; i < ref.length; i++) ref[i] /= kept;
    // Left display (already tonemapped) and our latest OIDN output.
    await renderFrame();
    renderToRT(leftDisplayRT, tmLeftQuad);
    const leftBytes = await readRT(leftDisplayRT);
    const out = await runOurDenoise();
    const rightBytes = out ? await readTex(out) : new Uint8ClampedArray(RES * RES * 4);
    psnrLeft = psnr(leftBytes, ref);
    psnrRight = psnr(rightBytes, ref);
    controls.autoRotate = wasMotion;
    refBtn.disabled = false; refBtn.textContent = 'Measure PSNR (hold camera still)';
    return { left: psnrLeft, right: psnrRight };
  }
  refBtn.addEventListener('click', () => { measurePSNR().catch((e) => log('PSNR error: ' + (e as Error).message)); });

  // --- main loop ---
  let frameId = 0;
  let leftMs = 0;
  async function renderFrame() {
    controls.update();
    const t0 = performance.now();
    renderToRT(leftLinearRT, heavyQuad);   // scene + SSR + temporalReproject + recurrentDenoise
    renderToRT(rawLinearRT, rawQuad);       // scene + RAW SSR (plain-texture composite)
    renderToRT(leftDisplayRT, tmLeftQuad);  // tonemap left -> display
    // Honest per-frame cost: wait for the GPU to finish this frame's work, the
    // same way the OIDN side's timing awaits its denoise (else we'd only be timing
    // CPU command submission and the left side would look ~100x faster than it is).
    await device.queue.onSubmittedWorkDone();
    leftMs = performance.now() - t0;
  }
  function updateHud() {
    const cad = cadence();
    hudLeft.innerHTML =
      `<b>three.js recurrentDenoise()</b>\n` +
      `frame (SSR+denoise): <b>${leftMs.toFixed(1)} ms</b>  ~${(1000 / Math.max(leftMs, 0.01)).toFixed(0)} fps\n` +
      (Number.isFinite(psnrLeft) ? `PSNR vs converged: <b>${psnrLeft.toFixed(1)} dB</b>` : `temporal · every frame`);
    hudRight.innerHTML =
      `<b>our OIDN full-frame</b>${denoiser.modelName ? ` · <b>${denoiser.modelName}</b>` : ''}\n` +
      `denoise: <b>${ourMs ? ourMs.toFixed(1) + ' ms' : '—'}</b>  cadence: every ${cad} frames\n` +
      (Number.isFinite(psnrRight) ? `PSNR vs converged: <b>${psnrRight.toFixed(1)} dB</b>` : `cadence-based · lags in motion`);
  }

  async function loop() {
    if ((window as unknown as Record<string, unknown>).__pauseLoop) { requestAnimationFrame(loop); return; }
    frameId++;
    await renderFrame();
    if (!headless) blit(backendGet(leftDisplayRT.texture)!, leftCtx);

    // our denoise on cadence (skip if a run is still in flight)
    if (!ourBusy && frameId % cadence() === 0) {
      ourBusy = true;
      runOurDenoise()
        .catch((e) => log('our denoise error: ' + (e as Error).message))
        .finally(() => { ourBusy = false; });
    }
    // right side shows the raw (tonemapped) frame until the first OIDN result lands
    if (!hasDenoised && !headless) { renderToRT(rawDisplayRT, tmRawQuad); blit(backendGet(rawDisplayRT.texture)!, rightCtx); }

    updateHud();
    requestAnimationFrame(loop);
  }
  loop();
  demoFooter('realtime-compare');
  log(`ready — SSR denoised two ways on one device (${RES}×${RES}/side)`);

  // --- headless verification hook ---
  (window as unknown as Record<string, unknown>).__verify = async () => {
    (window as unknown as Record<string, unknown>).__pauseLoop = true;
    await new Promise((r) => setTimeout(r, 50));
    // Warm up temporal history, then capture all three surfaces + a PSNR read.
    for (let i = 0; i < 24; i++) await renderFrame();
    renderToRT(leftDisplayRT, tmLeftQuad);
    renderToRT(rawDisplayRT, tmRawQuad);
    const out = await runOurDenoise();
    const leftB = await readRT(leftDisplayRT);
    const rawB = await readRT(rawDisplayRT);
    const rightB = out ? await readTex(out) : new Uint8ClampedArray(RES * RES * 4);
    const ps = await measurePSNR(60).catch(() => ({ left: NaN, right: NaN }));
    (window as unknown as Record<string, unknown>).__pauseLoop = false; // the paused loop resumes itself
    return {
      modelName: denoiser.modelName ?? null,
      ourMs, leftMs,
      left: { nonBlack: nonBlack(leftB), localVar: localVar(leftB) },
      right: { nonBlack: nonBlack(rightB), localVar: localVar(rightB) },
      raw: { nonBlack: nonBlack(rawB), localVar: localVar(rawB) },
      psnr: ps,
    };
  };
  // DEBUG: capture the surfaces as PNG data URLs for offline inspection.
  const toDataUrl = (b: Uint8ClampedArray): string => {
    const cv = document.createElement('canvas'); cv.width = RES; cv.height = RES;
    cv.getContext('2d')!.putImageData(new ImageData(b, RES, RES), 0, 0);
    return cv.toDataURL('image/png');
  };
  (window as unknown as Record<string, unknown>).__shot = async () => {
    (window as unknown as Record<string, unknown>).__pauseLoop = true;
    await new Promise((r) => setTimeout(r, 30));
    for (let i = 0; i < 32; i++) await renderFrame();
    renderToRT(leftDisplayRT, tmLeftQuad);
    renderToRT(rawDisplayRT, tmRawQuad);
    renderToRT(ssrDbgRT, tmSsrQuad);
    const out = await runOurDenoise();
    const left = toDataUrl(await readRT(leftDisplayRT));
    const raw = toDataUrl(await readRT(rawDisplayRT));
    const ssr = toDataUrl(await readRT(ssrDbgRT));
    const our = out ? toDataUrl(await readTex(out)) : null;
    (window as unknown as Record<string, unknown>).__pauseLoop = false; // the paused loop resumes itself
    return { left, raw, ssr, our };
  };
  (window as unknown as Record<string, unknown>).__app = { renderer, denoiser, scene, camera, denoiseNode, ssrNode };
}

main().catch((e) => log('ERROR: ' + (e as Error).message));
