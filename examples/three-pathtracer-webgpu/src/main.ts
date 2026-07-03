// Phase 2 — three r185 WebGPUPathTracer + the WebGPU/ONNX denoiser on ONE shared
// GPUDevice (ORT creates it; three.js borrows it — onnxruntime issue #26107).
//
// First cut: path-trace a small scene, then denoise the accumulated color via the
// `denoiser` package. Aux (albedo/normal) G-buffer + zero-copy GPU IO are the next
// increments; this validates the shared-device pipeline end to end.
import * as THREE from 'three/webgpu';
import { mrt, diffuseColor, normalView, texture } from 'three/tsl';
import { fsr1 } from 'three/addons/tsl/display/FSR1Node.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
import { GradientEquirectTexture } from 'three-gpu-pathtracer/src/textures/GradientEquirectTexture.js';
import { Denoiser } from 'denoiser';

const status = document.querySelector<HTMLPreElement>('#status')!;
const log = (m: string) => { status.textContent += m + '\n'; console.log(m); };

function buildScene(): { scene: THREE.Scene; camera: THREE.PerspectiveCamera } {
  const scene = new THREE.Scene();
  // Light via environment (the WebGPU path tracer doesn't support analytic lights yet).
  // GradientEquirectTexture (from the pathtracer) is configured for compute sampling,
  // unlike a raw DataTexture which trips three's compute sampler codegen.
  const env = new GradientEquirectTexture();
  env.topColor.set(0xbfd4ff);
  env.bottomColor.set(0x666666);
  env.update();
  scene.environment = env;
  scene.background = env;
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.set(0, 1.5, 4);
  camera.lookAt(0, 0.5, 0);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(20, 20),
    new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.8 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const colors = [0xe53e3e, 0x38a169, 0x3182ce];
  colors.forEach((c, i) => {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.6, 32, 32),
      new THREE.MeshStandardMaterial({ color: c, roughness: 0.2 + i * 0.3, metalness: i === 2 ? 1 : 0 }),
    );
    m.position.set((i - 1) * 1.6, 0.6, 0);
    scene.add(m);
  });

  // No analytic lights — the WebGPU path tracer lights via scene.environment (above).
  return { scene, camera };
}

// Patch requestDevice to request the adapter's MAX limits + all features, BEFORE any
// device is created. ORT (which creates the shared device first) otherwise requests a
// minimal device, and the path tracer's heavy compute pipeline then fails validation.
// (onnxruntime issue #26107 workaround — make the one shared device capable enough.)
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

async function main() {
  if (!('gpu' in navigator)) { log('ERROR: WebGPU not available.'); return; }
  patchWebGPUForMaxLimits();

  // 1) Denoiser first, so ORT owns the GPUDevice we then share with three.js.
  const denoiser = new Denoiser();
  denoiser.weightsUrl = '/models';
  // WebGPU render targets read bottom-up; flip so the denoised output is right-side up.
  denoiser.flipOutputY = true;
  await denoiser.build();
  const device = denoiser.device!;
  log(`denoiser ready; sharing GPUDevice with three.js: ${device ? 'yes' : 'no'}`);
  device.lost.then((info) => log(`DEVICE LOST: ${info.reason} — ${info.message}`));
  device.addEventListener('uncapturederror', (e) =>
    log(`UNCAPTURED GPU ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`));

  // 2) three.js WebGPURenderer on the SAME device.
  // FSR mode (?fsr=1) keeps the render/denoise at 512 and FSR1-upscales 2x to a
  // 1024 output — i.e. rendering 25% of the display pixels. (Rendering at a
  // reduced size instead is blocked upstream: this unreleased WebGPUPathTracer
  // branch wedges on setSize/renderScale and hangs renderSample at non-512
  // sizes, so the tracer must stay at its initial resolution.)
  const fsrMode = new URLSearchParams(location.search).has('fsr');
  const RES = 512;
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, device });
  await renderer.init();
  renderer.setSize(RES, RES, false);

  const { scene, camera } = buildScene();

  // 3) WebGPU path tracer.
  // useMegakernel() (re)creates the internal tracer AND wires the material into it
  // (the constructor's default tracer has no material -> "bsdfSample of null").
  const pathTracer = new WebGPUPathTracer(renderer);
  pathTracer.useMegakernel(true);
  // Show the full-res noisy frame immediately (skip the low-res/fade transition) so a
  // low sample count is actually noisy — that's the point of the denoise pass.
  pathTracer.dynamicLowRes = false;
  pathTracer.renderDelay = 0;
  pathTracer.setScene(scene, camera);
  log('path tracer initialized; accumulating samples...');

  // Orbit the camera and watch the denoiser keep up — camera changes reset the
  // accumulation and the live loop re-denoises as samples arrive. Aux mode
  // re-rasterizes the G-buffer for the new view automatically.
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.5, 0);
  controls.update(); // BEFORE attaching the listener — the initial update fires
  // 'change', and poking pathTracer.updateCamera() mid-setup wedges the tracer
  // (same fragile-init family as its setSize/renderScale bugs).
  controls.addEventListener('change', () => {
    pathTracer.updateCamera();
    gbufferRendered = false; // aux G-buffer is view-dependent
  });

  // Cap accumulation so the denoiser has noise to work with (modern GPUs blow past
  // hundreds of samples otherwise). Editable live; changing it restarts accumulation.
  const maxSamplesInput = document.querySelector<HTMLInputElement>('#maxSamples')!;
  const maxSamples = () => Math.max(1, parseInt(maxSamplesInput.value, 10) || 6);
  maxSamplesInput.addEventListener('change', () => pathTracer.reset());

  // The path tracer's live GPUTexture (float, linear HDR). Fetched fresh each use —
  // the tracer can replace its output target on reset/resize.
  const getTracerTexture = (): GPUTexture | undefined => {
    const target = pathTracer._pathTracer.outputTarget;
    const threeTexture = target.isTexture ? target : target.textures?.[0];
    return (renderer.backend as unknown as {
      get: (o: unknown) => { texture?: GPUTexture };
    }).get(threeTexture)?.texture;
  };

  // Progressive live denoise: the denoiser resolves straight into a three-owned
  // StorageTexture (caller-owned render target — no engine copy, no readback),
  // which a fullscreen quad then presents. pathtracer -> denoiser -> RT -> render.
  const liveCheckbox = document.querySelector<HTMLInputElement>('#live')!;
  // rgba16float target: the denoiser writes UNCLAMPED linear HDR into it and
  // three's own pipeline does the tonemapping + sRGB encode — no color handling
  // baked into the denoiser output. (srgb formats can't be STORAGE_BINDING.)
  // In FSR mode the denoiser instead writes tonemapped+sRGB (EASU's input
  // contract) and three's transforms are bypassed.
  const denoisedTex = new THREE.StorageTexture(RES, RES);
  denoisedTex.type = THREE.HalfFloatType;
  denoisedTex.generateMipmaps = false;
  denoisedTex.colorSpace = fsrMode ? THREE.NoColorSpace : THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.initTexture(denoisedTex);
  const denoisedGpuTex = (renderer.backend as unknown as {
    get: (o: unknown) => { texture?: GPUTexture };
  }).get(denoisedTex)?.texture;
  liveCheckbox.disabled = !denoisedGpuTex;

  // The compare overlay canvas: denoised output blitted over the raw path-traced
  // view, revealed by the slider (pure HTML/CSS clipping).
  const overlayCanvas = document.querySelector<HTMLCanvasElement>('#outGpu')!;
  const OVERLAY = fsrMode ? 1024 : 512; // FSR mode blits its 2x-upscaled result
  overlayCanvas.width = overlayCanvas.height = OVERLAY;
  const overlayCtx = overlayCanvas.getContext('webgpu')!;
  overlayCtx.configure({ device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  const blitToOverlay = (tex: GPUTexture) => {
    const enc = device.createCommandEncoder();
    enc.copyTextureToTexture({ texture: tex }, { texture: overlayCtx.getCurrentTexture() },
      { width: Math.min(tex.width, OVERLAY), height: Math.min(tex.height, OVERLAY) });
    device.queue.submit([enc.finish()]);
  };
  const reveal = document.querySelector<HTMLInputElement>('#reveal')!;
  const revealWrap = document.querySelector<HTMLDivElement>('#revealWrap')!;
  reveal.addEventListener('input', () => { revealWrap.style.width = `${reveal.value}%`; });
  // the one-shot buttons swap models/settings — keep them exclusive with live mode
  liveCheckbox.addEventListener('change', () => {
    document.querySelectorAll<HTMLButtonElement>('#denoise, #denoiseGpu')
      .forEach((b) => { b.disabled = liveCheckbox.checked; });
  });

  // G-buffer aux (TODO 2c): rasterize the SAME scene once into an MRT target —
  // albedo = material base color (unlit), normal = view-space normal [-1,1].
  // Rasterized aux is noise-free, so the cleanAux (calb_cnrm) models apply.
  const auxCheckbox = document.querySelector<HTMLInputElement>('#aux')!;
  const gbuffer = new THREE.RenderTarget(RES, RES, { count: 2, type: THREE.HalfFloatType });
  gbuffer.textures[0].name = 'albedo';
  gbuffer.textures[1].name = 'normal';
  let gbufferRendered = false;
  function renderGBuffer() {
    renderer.setMRT(mrt({ albedo: diffuseColor, normal: normalView }));
    renderer.setRenderTarget(gbuffer);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.setMRT(null);
    gbufferRendered = true;
  }
  const backendGet = (o: unknown) => (renderer.backend as unknown as {
    get: (o: unknown) => { texture?: GPUTexture };
  }).get(o)?.texture;

  // FSR1 mode (?fsr=1, page reload): path-trace + denoise at HALF resolution and
  // upscale 2x with three's official FSR1 (EASU+RCAS) TSL node into the #outGpu
  // canvas. Per AMD guidance EASU wants tonemapped, gamma-encoded, anti-aliased
  // (= denoised!) [0,1] input — the denoiser resolves with tonemapOutput
  // (ACES+sRGB) and the FSR chain passes values through untouched.
  const fsrCheckbox = document.querySelector<HTMLInputElement>('#fsr')!;
  fsrCheckbox.checked = fsrMode;
  fsrCheckbox.addEventListener('change', () => {
    location.search = fsrCheckbox.checked ? '?fsr=1' : '';
  });
  let renderFsr: (() => void) | undefined;
  if (fsrMode) {
    const OUT = 1024;
    const fsrNode = fsr1(texture(denoisedTex), 0.2);
    // FSR1Node auto-sizes its output to the renderer's drawing buffer (256 here);
    // pin it to the display resolution instead.
    const origSetSize = fsrNode.setSize.bind(fsrNode);
    fsrNode.setSize = () => origSetSize(OUT, OUT);
    const fsrTarget = new THREE.RenderTarget(OUT, OUT, { depthBuffer: false }); // rgba8unorm
    const fsrQuad = new THREE.QuadMesh(new THREE.NodeMaterial());
    (fsrQuad.material as THREE.NodeMaterial).fragmentNode = fsrNode; // raw write, no transforms
    renderFsr = () => {
      renderer.setRenderTarget(fsrTarget);
      fsrQuad.render(renderer);
      renderer.setRenderTarget(null);
      const src = backendGet(fsrTarget.texture);
      if (src) blitToOverlay(src); // 1024 FSR result onto the compare overlay
    };
  }

  let denoisingBusy = false;
  let lastDenoisedSample = -1;
  let liveMs = 0;

  async function runLiveDenoise() {
    const tracerTex = getTracerTexture();
    if (!tracerTex || !denoisedGpuTex) throw new Error('live denoise: textures unavailable');
    if (tracerTex.width !== RES) return; // tracer target not ready yet
    if (auxCheckbox.checked) {
      if (!gbufferRendered) renderGBuffer();
      const albedoTex = backendGet(gbuffer.textures[0]);
      const normalTex = backendGet(gbuffer.textures[1]);
      if (!albedoTex || !normalTex) throw new Error('aux: G-buffer textures unavailable');
      denoiser.setInputTexture('albedo', albedoTex);
      denoiser.setInputTexture('normal', normalTex);
    } else if (denoiser.props.useAlbedo) {
      denoiser.resetInputs(); // drop aux -> back to the color-only model
    }
    denoiser.hdr = true; // linear-HDR float input -> hdr model
    denoiser.srgb = false;
    denoiser.flipInputY = true; // bottom-up render target -> top-down output
    denoiser.flipOutputY = false;
    // Both modes: display-encoded output (ACES+sRGB in the resolve kernel),
    // top-down (flipInputY normalizes the bottom-up render target on read).
    denoiser.tonemapOutput = true;
    denoiser.outputMode = 'gpuTexture';
    denoiser.setInputTexture('color', tracerTex);
    if (fsrMode) {
      // resolve into the three-owned texture that feeds the fsr1() node
      denoiser.setOutputTexture(denoisedGpuTex);
      await denoiser.execute();
      renderFsr?.(); // EASU/RCAS 2x -> compare overlay
    } else {
      // engine-owned rgba8 texture, blitted straight onto the compare overlay
      denoiser.setOutputTexture(undefined);
      const outTex = (await denoiser.execute()) as GPUTexture;
      blitToOverlay(outTex);
    }
  }

  (window as unknown as Record<string, unknown>).__app = { pathTracer, renderer, denoiser, scene, camera };

  const loop = () => {
    const s = Math.floor(pathTracer.samples ?? 0);
    if (s < maxSamples()) pathTracer.renderSample();
    if (liveCheckbox.checked && !denoisingBusy && s !== lastDenoisedSample) {
      denoisingBusy = true;
      const t0 = performance.now();
      runLiveDenoise()
        .then(() => { liveMs = performance.now() - t0; lastDenoisedSample = s; })
        .catch((e) => { log('live denoise ERROR: ' + (e as Error).message); liveCheckbox.checked = false; })
        .finally(() => { denoisingBusy = false; });
    }
    // presentation happens inside runLiveDenoise (blit / FSR onto the overlay);
    // the main canvas always shows the raw accumulation for the compare slider.
    status.textContent = status.textContent!.replace(/samples:.*$/m, '').trimEnd() +
      `\nsamples: ${s} / ${maxSamples()}` +
      (liveCheckbox.checked && liveMs ? ` | live denoise: ${liveMs.toFixed(1)} ms` : '');
    requestAnimationFrame(loop);
  };
  loop();

  // 4) CPU-path denoise button (the old way, kept for comparison): float readback,
  // JS tonemap to LDR RGBA8, denoise with the ldr model, putImageData.
  const btn = document.querySelector<HTMLButtonElement>('#denoise')!;
  const outCanvas = document.querySelector<HTMLCanvasElement>('#out')!;
  btn.disabled = false;
  btn.addEventListener('click', async () => {
    log(`denoising (CPU path) at ${Math.floor(pathTracer.samples ?? 0)} samples...`);
    // Read the path tracer's linear-HDR float output target (drawImage on a WebGPU
    // canvas yields black), then tonemap (ACES) + sRGB-encode to match the display.
    const target = pathTracer._pathTracer.outputTarget;
    const w = target.width as number;
    const h = target.height as number;
    const t0 = performance.now();
    const stub = { textures: [target] };
    const linear = (await renderer.readRenderTargetPixelsAsync(stub, 0, 0, w, h)) as Float32Array;

    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      for (let c = 0; c < 3; c++) {
        let v = linear[i * 4 + c];
        // Narkowicz ACES approximation (≈ three's ACESFilmicToneMapping)
        v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
        v = Math.min(1, Math.max(0, v));
        v = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055; // linear->sRGB
        rgba[i * 4 + c] = v * 255;
      }
      rgba[i * 4 + 3] = 255;
    }
    const img = new ImageData(rgba, w, h);

    denoiser.hdr = false; // tonemapped LDR input -> ldr model
    denoiser.srgb = false;
    denoiser.flipOutputY = true; // render target rows arrive bottom-up
    denoiser.flipInputY = false;
    denoiser.tonemapOutput = false;
    denoiser.outputMode = 'imgData';
    denoiser.setCanvas(outCanvas);
    await denoiser.execute(img);
    log(`CPU path: denoised in ${(performance.now() - t0).toFixed(1)} ms (${w}×${h}, incl. readback+tonemap)`);
  });

  // 5) Zero-copy GPU path (one-shot): hand the path tracer's float StorageTexture
  // straight to the denoiser (real linear-HDR input, hdr model), get an rgba8
  // texture back, and blit it onto the compare overlay. No CPU pixels anywhere.
  const btnGpu = document.querySelector<HTMLButtonElement>('#denoiseGpu')!;
  btnGpu.disabled = false;
  btnGpu.addEventListener('click', async () => {
    log(`denoising (zero-copy GPU path) at ${Math.floor(pathTracer.samples ?? 0)} samples...`);
    const gpuTexture = getTracerTexture();
    if (!gpuTexture) { log('ERROR: could not resolve the render target’s GPUTexture'); return; }

    const t0 = performance.now();
    denoiser.hdr = true; // real linear-HDR floats -> hdr model
    denoiser.srgb = false;
    denoiser.flipInputY = true; // WebGPU render targets read bottom-up
    denoiser.flipOutputY = false;
    denoiser.tonemapOutput = true; // ACES + sRGB in the resolve kernel for display
    denoiser.outputMode = 'gpuTexture';
    denoiser.setOutputTexture(undefined); // engine-owned texture (we blit it below)
    denoiser.setInputTexture('color', gpuTexture);
    const outTex = (await denoiser.execute()) as GPUTexture;
    blitToOverlay(outTex);
    log(`GPU path: denoised in ${(performance.now() - t0).toFixed(1)} ms (${outTex.width}×${outTex.height}, zero-copy)`);
  });
}

main().catch((e) => log('ERROR: ' + (e as Error).message));
