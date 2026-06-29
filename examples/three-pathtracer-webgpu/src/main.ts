// Phase 2 — three r185 WebGPUPathTracer + the WebGPU/ONNX denoiser on ONE shared
// GPUDevice (ORT creates it; three.js borrows it — onnxruntime issue #26107).
//
// First cut: path-trace a small scene, then denoise the accumulated color via the
// `denoiser` package. Aux (albedo/normal) G-buffer + zero-copy GPU IO are the next
// increments; this validates the shared-device pipeline end to end.
import * as THREE from 'three/webgpu';
import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
import { Denoiser } from 'denoiser';

const status = document.querySelector<HTMLPreElement>('#status')!;
const log = (m: string) => { status.textContent += m + '\n'; console.log(m); };

function buildScene(): { scene: THREE.Scene; camera: THREE.PerspectiveCamera } {
  const scene = new THREE.Scene();
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

  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.position.set(3, 5, 2);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xffffff, 0.3));
  return { scene, camera };
}

async function main() {
  if (!('gpu' in navigator)) { log('ERROR: WebGPU not available.'); return; }

  // 1) Denoiser first, so ORT owns the GPUDevice we then share with three.js.
  const denoiser = new Denoiser();
  denoiser.weightsUrl = '/models';
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
  const pathTracer = new WebGPUPathTracer(renderer);
  pathTracer.setScene(scene, camera);
  log('path tracer initialized; accumulating samples...');

  let raf = 0;
  const loop = () => {
    pathTracer.renderSample();
    status.textContent = status.textContent!.replace(/samples:.*$/m, '') + `samples: ${Math.floor(pathTracer.samples ?? 0)}`;
    raf = requestAnimationFrame(loop);
  };
  loop();

  // 4) Denoise button: read back the accumulated frame and run it through the denoiser.
  const btn = document.querySelector<HTMLButtonElement>('#denoise')!;
  const outCanvas = document.querySelector<HTMLCanvasElement>('#out')!;
  btn.disabled = false;
  btn.addEventListener('click', async () => {
    cancelAnimationFrame(raf);
    log(`denoising at ${Math.floor(pathTracer.samples ?? 0)} samples...`);
    const w = 512, h = 512;
    const buf = new Uint8Array(w * h * 4);
    await renderer.readRenderTargetPixelsAsync(pathTracer.target, 0, 0, w, h, buf);
    const img = new ImageData(new Uint8ClampedArray(buf.buffer), w, h);
    denoiser.setCanvas(outCanvas);
    const t0 = performance.now();
    await denoiser.execute(img);
    log(`denoised in ${(performance.now() - t0).toFixed(1)} ms`);
    loop();
  });
}

main().catch((e) => log('ERROR: ' + (e as Error).message));
