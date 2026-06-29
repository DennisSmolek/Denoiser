// Phase 2 — three r185 WebGPUPathTracer + the WebGPU/ONNX denoiser on ONE shared
// GPUDevice (ORT creates it; three.js borrows it — onnxruntime issue #26107).
//
// First cut: path-trace a small scene, then denoise the accumulated color via the
// `denoiser` package. Aux (albedo/normal) G-buffer + zero-copy GPU IO are the next
// increments; this validates the shared-device pipeline end to end.
import * as THREE from 'three/webgpu';
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

  // 2) three.js WebGPURenderer on the SAME device.
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, device });
  await renderer.init();
  renderer.setSize(512, 512, false);

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

  // Cap accumulation so the denoiser has noise to work with (modern GPUs blow past
  // hundreds of samples otherwise). Editable live; changing it restarts accumulation.
  const maxSamplesInput = document.querySelector<HTMLInputElement>('#maxSamples')!;
  const maxSamples = () => Math.max(1, parseInt(maxSamplesInput.value, 10) || 6);
  maxSamplesInput.addEventListener('change', () => pathTracer.reset());

  const loop = () => {
    if (Math.floor(pathTracer.samples ?? 0) < maxSamples()) pathTracer.renderSample();
    status.textContent = status.textContent!.replace(/samples:.*$/m, '').trimEnd() +
      `\nsamples: ${Math.floor(pathTracer.samples ?? 0)} / ${maxSamples()}`;
    requestAnimationFrame(loop);
  };
  loop();

  // 4) Denoise button: grab the path tracer's (tonemapped) canvas pixels and run them
  // through the denoiser package. drawImage works across canvas context types, so this
  // sidesteps the float StorageTexture readback entirely.
  const btn = document.querySelector<HTMLButtonElement>('#denoise')!;
  const outCanvas = document.querySelector<HTMLCanvasElement>('#out')!;
  btn.disabled = false;
  btn.addEventListener('click', async () => {
    log(`denoising at ${Math.floor(pathTracer.samples ?? 0)} samples...`);
    // Read the path tracer's linear-HDR float output target (drawImage on a WebGPU
    // canvas yields black), then tonemap (ACES) + sRGB-encode to match the display.
    const target = pathTracer._pathTracer.outputTarget;
    const w = target.width as number;
    const h = target.height as number;
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

    denoiser.setCanvas(outCanvas);
    const t0 = performance.now();
    await denoiser.execute(img);
    log(`denoised in ${(performance.now() - t0).toFixed(1)} ms (${w}×${h})`);
  });
}

main().catch((e) => log('ERROR: ' + (e as Error).message));
