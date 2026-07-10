// LDraw Eiffel Tower demo: a table, an HDRI, a gelatinous cube, and the LEGO
// Architecture Eiffel Tower (21019) — orbit with camera-controls and you get
// the plain raster render; let the camera rest and the WebGPU path tracer
// accumulates, then the denoiser resolves the noise-free frame on top.
//
// Same shared-GPUDevice setup as examples/three-pathtracer-webgpu: ORT (the
// denoiser) creates the device first, three.js borrows it (onnxruntime #26107).
import * as THREE from 'three/webgpu';
import { mrt, diffuseColor, normalView, texture as textureNode, uv, vec2, vec4 } from 'three/tsl';
import CameraControls from 'camera-controls';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { LDrawLoader } from 'three/addons/loaders/LDrawLoader.js';
import { LDrawConditionalLineMaterial } from 'three/addons/materials/LDrawConditionalLineNodeMaterial.js';
import { LDrawUtils } from 'three/addons/utils/LDrawUtils.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
import { Denoiser } from 'denoiser';
import { installGalleryCapture } from '../../_shared/gallery-capture';

const status = document.querySelector<HTMLPreElement>('#status')!;
const modeLabel = document.querySelector<HTMLSpanElement>('#mode')!;
const log = (m: string) => { status.textContent += m + '\n'; console.log(m); };

// The upstream WebGPUPathTracer branch wedges on setSize/renderScale — the
// tracer must stay at its initial 512 resolution (see three-pathtracer-webgpu).
const RES = 512;

async function buildScene(): Promise<{ scene: THREE.Scene; camera: THREE.PerspectiveCamera }> {
  const scene = new THREE.Scene();

  // Soft overcast sky: near-uniform luminance. HDRIs with small intense
  // lights (studio lamps, sun disks) stay salt-and-pepper noisy in this
  // WebGPU tracer branch even at high sample counts — concentrated-source
  // sampling isn't there yet — and that speckle survives the denoiser.
  // BASE_URL keeps assets resolving under the deployed subpath (/denoiser/ldraw-eiffel/); it is "/" in dev.
  const env = await new HDRLoader().loadAsync(`${import.meta.env.BASE_URL}assets/kloppenheim_06_puresky_1k.hdr`);
  env.mapping = THREE.EquirectangularReflectionMapping;
  scene.environment = env;
  scene.background = env;

  // Table surface.
  const table = new THREE.Mesh(
    new RoundedBoxGeometry(11, 0.5, 11, 3, 0.08),
    // Rough enough that the glossy lobe doesn't turn the HDRI's small bright
    // lamps into fireflies (the WebGPU tracer has no glossy filter yet).
    new THREE.MeshStandardMaterial({ color: 0x9a6a39, roughness: 0.7 }),
  );
  table.name = 'table';
  table.position.y = -0.25;
  scene.add(table);

  // Gelatinous cube. Two looks, one material: `transmission` gives the raster
  // preview real glassiness; the traced side of the WebGPU branch has no
  // refraction yet, so `transparent`+`opacity` drives its stochastic
  // pass-through there instead.
  const gel = new THREE.Mesh(
    new THREE.BoxGeometry(1.3, 1.3, 1.3),
    new THREE.MeshPhysicalMaterial({
      color: 0x3ec46d,
      roughness: 0.18,
      transmission: 1,
      ior: 1.35,
      thickness: 1.2,
      transparent: true,
      opacity: 0.45,
    }),
  );
  gel.name = 'gel';
  gel.position.set(2.4, 0.651, 1.6);
  gel.rotation.y = 0.5;
  scene.add(gel);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
  camera.position.set(6.5, 4.5, 8.5);
  return { scene, camera };
}

async function loadEiffel(scene: THREE.Scene) {
  const t0 = performance.now();
  const loader = new LDrawLoader();
  loader.smoothNormals = true;
  // r185 requires the renderer-appropriate conditional-line material to be
  // injected (node material for WebGPU). The lines get stripped below anyway.
  loader.setConditionalLineMaterial(LDrawConditionalLineMaterial);
  // Packed MPD (tools/pack-ldraw.mjs): every part + LDConfig colors inlined,
  // so no runtime trips to a parts-library CDN.
  const raw = await loader.loadAsync(`${import.meta.env.BASE_URL}assets/eiffel-tower_Packed.mpd`);
  raw.rotation.x = Math.PI; // LDraw is -Y up
  raw.updateMatrixWorld(true);

  // One mesh with material groups instead of ~1000 part meshes — one BLAS
  // build instead of hundreds (the WebGPU tracer supports geometry groups).
  const model = LDrawUtils.mergeObject(raw);

  // Drop the LDraw construction/edge lines: the tracer only sees meshes, and
  // the raster preview should match what gets traced.
  const lines: THREE.Object3D[] = [];
  model.traverse((c: THREE.Object3D) => {
    if ((c as THREE.LineSegments).isLineSegments || (c as THREE.Line).isLine) lines.push(c);
  });
  lines.forEach((l) => l.removeFromParent());

  // Scale to ~3.6 units tall, centered, base resting on the table (y=0).
  let tris = 0;
  let meshCount = 0;
  model.traverse((c: THREE.Mesh) => {
    if (c.isMesh) { meshCount++; tris += (c.geometry.index?.count ?? c.geometry.attributes.position.count) / 3; }
  });
  log(`merged model: ${meshCount} mesh(es), ${tris.toFixed(0)} tris`);
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  log(`model size: ${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)}`);
  model.scale.setScalar(3.6 / size.y);
  model.updateMatrixWorld(true);
  box.setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  model.position.set(-center.x, model.position.y - box.min.y, -center.z);
  model.name = 'eiffel';
  scene.add(model);
  log(`Eiffel Tower loaded + merged in ${(performance.now() - t0).toFixed(0)} ms`);
}

// Request the adapter's MAX limits/features BEFORE any device exists: ORT
// creates the shared device and would otherwise request a minimal one that
// fails the path tracer's compute pipeline validation (see the other example).
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
  // Dev serves converted models from /models (vite middleware); prod falls back to the CDN default.
  const denoiser = await Denoiser.create({ weightsUrl: import.meta.env.DEV ? '/models' : undefined });
  const device = denoiser.device;
  log('denoiser ready; sharing its GPUDevice with three.js');
  device.lost.then((info) => log(`DEVICE LOST: ${info.reason} — ${info.message}`));
  device.addEventListener('uncapturederror', (e) =>
    log(`UNCAPTURED GPU ERROR: ${(e as GPUUncapturedErrorEvent).error.message}`));

  // 2) three.js WebGPURenderer on the SAME device.
  const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, device });
  // three's official Inspector (r180+): render-target viewer, node parameters,
  // profiler. Opt-in via ?inspector — it overlays its own UI.
  if (new URLSearchParams(location.search).has('inspector')) {
    const { Inspector } = await import('three/addons/inspector/Inspector.js');
    renderer.inspector = new Inspector();
    log('three.js Inspector attached');
  }
  await renderer.init();
  renderer.setSize(RES, RES, false);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;

  // 3) Scene + model.
  const { scene, camera } = await buildScene();
  await loadEiffel(scene);

  // 4) Path tracer (megakernel; full-res noisy frames from sample 1).
  const t0 = performance.now();
  const pathTracer = new WebGPUPathTracer(renderer);
  pathTracer.useMegakernel(true);
  pathTracer.dynamicLowRes = false;
  pathTracer.renderDelay = 0;
  pathTracer.setScene(scene, camera);
  log(`path tracer BVH built in ${(performance.now() - t0).toFixed(0)} ms`);

  // 5) camera-controls: raster preview while the camera moves, trace at rest.
  // camera-controls is fully typed but `three` here isn't (repo status quo) — cast.
  CameraControls.install({ THREE: THREE as unknown as Parameters<typeof CameraControls.install>[0]['THREE'] });
  const controls = new CameraControls(camera, canvas);
  controls.minDistance = 2;
  controls.maxDistance = 50;
  controls.setLookAt(6.5, 4.5, 8.5, 0, 1.6, 0, false);
  let userActive = false;
  controls.addEventListener('controlstart', () => { userActive = true; });
  controls.addEventListener('controlend', () => { userActive = false; });

  // Live GPUTexture of the tracer's accumulation (refetched — it can be
  // replaced on reset) and of three-owned render targets.
  const backendGet = (o: unknown) => (renderer.backend as unknown as {
    get: (o: unknown) => { texture?: GPUTexture };
  }).get(o)?.texture;
  const getTracerTexture = (): GPUTexture | undefined => {
    const target = pathTracer._pathTracer.outputTarget;
    const threeTexture = target.isTexture ? target : target.textures?.[0];
    return backendGet(threeTexture);
  };

  // Denoised overlay canvas (webgpu context, plain texture copy). ?headless=1
  // skips it — a second getContext('webgpu')+configure stalls headless Chrome
  // (same as three-pathtracer-webgpu). The gallery capture doesn't need it.
  const headless = new URLSearchParams(location.search).has('headless');
  const overlayCanvas = document.querySelector<HTMLCanvasElement>('#denoised')!;
  const overlayCtx = headless ? undefined : overlayCanvas.getContext('webgpu')!;
  overlayCtx?.configure({
    device, format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const blitToOverlay = (tex: GPUTexture) => {
    if (!overlayCtx) return;
    const enc = device.createCommandEncoder();
    enc.copyTextureToTexture({ texture: tex }, { texture: overlayCtx.getCurrentTexture() },
      { width: Math.min(tex.width, RES), height: Math.min(tex.height, RES) });
    device.queue.submit([enc.finish()]);
  };
  const reveal = document.querySelector<HTMLInputElement>('#reveal')!;
  const revealWrap = document.querySelector<HTMLDivElement>('#revealWrap')!;
  reveal.addEventListener('input', () => { revealWrap.style.width = `${reveal.value}%`; });

  // G-buffer aux: rasterize the same view once into an MRT target — albedo =
  // unlit base color, normal = view-space normal. Noise-free aux → cleanAux models.
  const auxCheckbox = document.querySelector<HTMLInputElement>('#aux')!;
  const auxOpaqueCheckbox = document.querySelector<HTMLInputElement>('#auxOpaque')!;
  // Two separate passes, because the two guides want different scene state
  // (OIDN conventions):
  //  - albedo: background ON — env color IS the correct albedo for env pixels
  //    (clamped engine-side); transparent surfaces through-blend, unless the
  //    "opaque aux" toggle forces first-hit albedo.
  //  - normal: background OFF — the env quad's normalView is a meaningless
  //    gradient, while a cleared target reads as normal=0, exactly OIDN's env
  //    convention. Transparency is ALWAYS off here: alpha-blended normals are
  //    garbage; first-hit normals are the contract.
  const albedoRT = new THREE.RenderTarget(RES, RES, { type: THREE.HalfFloatType });
  const normalRT = new THREE.RenderTarget(RES, RES, { type: THREE.HalfFloatType });
  // MRTNode routes outputs to render-target textures BY NAME — unnamed
  // textures make the index lookup return -1 and the node build crash.
  albedoRT.texture.name = 'albedo';
  normalRT.texture.name = 'normal';
  let gbufferRendered = false;
  function forceOpaque(): Array<() => void> {
    const restore: Array<() => void> = [];
    scene.traverse((c: THREE.Mesh) => {
      const mats: THREE.Material[] = c.isMesh
        ? (Array.isArray(c.material) ? c.material : [c.material])
        : [];
      for (const m of mats) {
        if (m.transparent) {
          m.transparent = false;
          m.needsUpdate = true;
          restore.push(() => { m.transparent = true; m.needsUpdate = true; });
        }
      }
    });
    return restore;
  }
  function renderGBuffer() {
    // albedo pass
    let restore: Array<() => void> = auxOpaqueCheckbox.checked ? forceOpaque() : [];
    try {
      renderer.setMRT(mrt({ albedo: diffuseColor }));
      renderer.setRenderTarget(albedoRT);
      renderer.render(scene, camera);
    } finally {
      // ALWAYS unwind — leaked MRT state crashes every later render.
      renderer.setRenderTarget(null);
      renderer.setMRT(null);
      restore.forEach((fn) => fn());
    }
    // normal pass
    restore = forceOpaque();
    const bg = scene.background;
    scene.background = null;
    try {
      renderer.setMRT(mrt({ normal: normalView }));
      renderer.setRenderTarget(normalRT);
      renderer.render(scene, camera);
      gbufferRendered = true;
    } finally {
      renderer.setRenderTarget(null);
      renderer.setMRT(null);
      scene.background = bg;
      restore.forEach((fn) => fn());
    }
  }

  // UI knobs.
  const maxSamplesInput = document.querySelector<HTMLInputElement>('#maxSamples')!;
  const maxSamples = () => Math.max(1, parseInt(maxSamplesInput.value, 10) || 32);
  const progressiveCheckbox = document.querySelector<HTMLInputElement>('#progressive')!;
  const qualitySel = document.querySelector<HTMLSelectElement>('#quality')!;
  denoiser.quality = qualitySel.value as 'fast' | 'balanced'; // honor the HTML default

  let denoiseBusy = false;
  let denoisedAtSample = -1;
  let denoiseMs = 0;

  const invalidateDenoise = () => {
    denoisedAtSample = -1;
    denoiser.abort();
    overlayCanvas.style.opacity = '0';
  };
  qualitySel.addEventListener('change', () => {
    denoiser.quality = qualitySel.value as 'fast' | 'balanced';
    invalidateDenoise();
  });
  auxCheckbox.addEventListener('change', () => { gbufferRendered = false; invalidateDenoise(); });
  auxOpaqueCheckbox.addEventListener('change', () => { gbufferRendered = false; invalidateDenoise(); });
  maxSamplesInput.addEventListener('change', () => { pathTracer.reset(); invalidateDenoise(); });

  // --- input debug views: put the network's actual inputs on the overlay ---
  // color is shown WITH the flip the engine applies (inputFlipY) and the aux
  // as-is (auxInputFlipY: false), i.e. exactly as the network pairs them — a
  // wrong flip flag shows up here as color/aux vertical mismatch.
  const viewSel = document.querySelector<HTMLSelectElement>('#viewMode')!;
  const debugRT = new THREE.RenderTarget(RES, RES); // rgba8unorm
  const debugQuad = new THREE.QuadMesh(new THREE.NodeMaterial());
  const getTracerThreeTexture = () => {
    const target = pathTracer._pathTracer.outputTarget;
    return target.isTexture ? target : target.textures?.[0];
  };
  let debugKey = '';
  function renderDebugView(view: string) {
    if (view !== 'color' && !gbufferRendered) renderGBuffer();
    const src = view === 'color' ? getTracerThreeTexture()
      : view === 'albedo' ? albedoRT.texture : normalRT.texture;
    if (!src) return;
    const key = `${view}:${src.uuid}`;
    if (key !== debugKey) {
      const t = view === 'color'
        ? textureNode(src, vec2(uv().x, uv().y.oneMinus())) // engine's inputFlipY
        : textureNode(src);
      // Display transforms only: color = linear HDR -> Reinhard + gamma;
      // albedo = linear [0,1] -> gamma; normal = [-1,1] -> [0,1] RAW (no
      // gamma) so wrong ranges read as washed-out gray or hard clipping.
      const rgb = view === 'color' ? t.rgb.div(t.rgb.add(1)).pow(1 / 2.2)
        : view === 'albedo' ? t.rgb.pow(1 / 2.2)
          : t.rgb.mul(0.5).add(0.5);
      (debugQuad.material as THREE.NodeMaterial).fragmentNode = vec4(rgb, 1);
      (debugQuad.material as THREE.NodeMaterial).needsUpdate = true;
      debugKey = key;
    }
    try {
      renderer.setRenderTarget(debugRT);
      debugQuad.render(renderer);
    } finally {
      renderer.setRenderTarget(null);
    }
    const gpu = backendGet(debugRT.texture);
    if (gpu) blitToOverlay(gpu);
  }
  viewSel.addEventListener('change', () => {
    if (viewSel.value === 'result') invalidateDenoise(); // fresh denoise re-blits the overlay
  });

  async function runDenoise(samples: number) {
    const colorTex = getTracerTexture();
    if (!colorTex || colorTex.width !== RES) return; // tracer target not ready
    let albedo: GPUTexture | undefined;
    let normal: GPUTexture | undefined;
    if (auxCheckbox.checked) {
      if (!gbufferRendered) renderGBuffer();
      albedo = backendGet(albedoRT.texture);
      normal = backendGet(normalRT.texture);
      if (!albedo || !normal) throw new Error('aux: G-buffer textures unavailable');
    }
    const t = performance.now();
    // Linear-HDR tracer input (bottom-up), raster aux already top-down,
    // display-encoded (ACES+sRGB) output to match the raster preview.
    const out = await denoiser.denoiseTextures({
      color: colorTex,
      albedo, normal,
      hdr: true,
      inputFlipY: true,
      auxInputFlipY: false,
      transfer: 'aces-srgb',
    });
    if (!out) return; // aborted mid-flight — the camera moved again
    if (viewSel.value === 'result') { // an input debug view owns the overlay otherwise
      blitToOverlay(out);
      overlayCanvas.style.opacity = '1';
    }
    denoiseMs = performance.now() - t;
    denoisedAtSample = samples;
  }

  // 6) The mode loop: camera moving → raster preview; at rest → accumulate
  // samples, then denoise (each sample when "progressive", else once at the end).
  type Mode = 'preview' | 'trace';
  let mode: Mode = 'preview';
  let cameraDirty = false; // only reset the tracer when the camera actually moved
  const clock = new THREE.Clock();

  (window as unknown as Record<string, unknown>).__app =
    { pathTracer, renderer, denoiser, scene, camera, controls, albedoRT, normalRT, backendGet, renderGBuffer };

  // Gallery-asset capture (Phase B B1.3): spp ladder + reference + albedo/normal
  // AOVs for the gallery demo (tools/capture-gallery, or ?capture=1 headed button).
  installGalleryCapture({
    THREE, tsl: { mrt, diffuseColor, normalView, texture: textureNode, vec4 },
    renderer, device, pathTracer, scene, camera,
    getTracerTexture, backendGet, res: RES,
    sceneId: 'eiffel', title: 'LDraw Eiffel Tower',
    log,
  });

  let loggedLoopError = false;
  const loop = () => {
    requestAnimationFrame(loop);
    if ((window as unknown as Record<string, unknown>).__capturing) return;
    try {
      loopBody();
    } catch (e) {
      if (!loggedLoopError) { loggedLoopError = true; log('LOOP ERROR: ' + ((e as Error).stack ?? e)); }
    }
  };
  const loopBody = () => {
    const updated = controls.update(clock.getDelta());
    const moving = updated || userActive;

    if (moving) {
      if (mode !== 'preview') {
        mode = 'preview';
        invalidateDenoise();
      }
      cameraDirty = true;
      gbufferRendered = false;
      renderer.render(scene, camera);
    } else {
      if (mode !== 'trace') {
        mode = 'trace';
        if (cameraDirty) {
          pathTracer.updateCamera(); // resets accumulation for the settled view
          cameraDirty = false;
        }
      }
      const s = Math.floor(pathTracer.samples ?? 0);
      if (s < maxSamples()) pathTracer.renderSample();
      const shouldDenoise = progressiveCheckbox.checked
        ? s > 0 && s !== denoisedAtSample
        : s >= maxSamples() && denoisedAtSample !== s;
      if (shouldDenoise && !denoiseBusy) {
        denoiseBusy = true;
        runDenoise(s)
          .catch((e) => log('denoise ERROR: ' + ((e as Error).stack ?? (e as Error).message)))
          .finally(() => { denoiseBusy = false; });
      }
    }

    // Input debug view: refresh every frame (tracks orbit + new samples).
    if (viewSel.value !== 'result') {
      renderDebugView(viewSel.value);
      overlayCanvas.style.opacity = '1';
    }

    const s = Math.floor(pathTracer.samples ?? 0);
    modeLabel.textContent = viewSel.value !== 'result'
      ? `input: ${viewSel.value}`
      : mode === 'preview'
        ? 'raster preview'
        : s < maxSamples()
          ? `path tracing ${s}/${maxSamples()}`
          : denoisedAtSample >= 0 ? 'denoised' : 'denoising…';
    status.textContent = status.textContent!.replace(/samples:.*$/m, '').trimEnd() +
      `\nsamples: ${s} / ${maxSamples()}` +
      (denoiseMs ? ` | denoise: ${denoiseMs.toFixed(1)} ms` : '');
  };
  loop();

  // rAF suspends while the page is hidden — say so instead of looking wedged.
  document.addEventListener('visibilitychange', () => {
    log(document.hidden ? 'PAUSED — page hidden (rAF suspended)' : 'resumed');
  });
}

main().catch((e) => log('ERROR: ' + (e as Error).message));
