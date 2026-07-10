// Shared gallery-asset capture for the pathtracer examples (Phase B, B1.3).
//
// Exposes `window.__captureGallery()` — accumulates the path tracer over an spp
// ladder and produces, per scene, the exact assets the gallery demo consumes:
//   - noisy color at 1/2/4/8/16 spp, ACES-tonemapped + sRGB (the display transform)
//   - a converged reference frame (high spp)
//   - albedo AOV as the network expects it (linear [0,1] bytes)
//   - normal AOV encoded [-1,1] -> [0,1] (OIDN conventions: env hits = normal 0,
//     first-hit, opaque — rendered here with the background OFF so env pixels read
//     as a flat 0.5 gray, NOT the env quad's normalView gradient that was the old
//     aux bug).
//
// Drive it headlessly with tools/capture-gallery, or add `?capture=1` to the page
// for a manual "capture gallery assets" button (headed fallback that downloads the
// PNGs). Both call the same `window.__captureGallery`.
//
// Imported via a RELATIVE path, like ../../_shared/chrome. The example passes in its
// own THREE + TSL nodes so there's exactly one three instance (the pathtracer needs
// this — see each example's vite.config.ts dedupe note).

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface GalleryCaptureDeps {
  THREE: any; // typeof import('three/webgpu')
  tsl: {
    mrt: any;
    diffuseColor: any;
    normalView: any;
    texture: any;
    vec4: any;
  };
  renderer: any; // THREE.WebGPURenderer
  device: GPUDevice;
  pathTracer: any; // WebGPUPathTracer
  scene: any; // THREE.Scene
  camera: any; // THREE.Camera
  /** Live GPUTexture of the tracer's accumulation (refetched — can be replaced on reset). */
  getTracerTexture: () => GPUTexture | undefined;
  /** three RenderTarget texture -> its backing GPUTexture. */
  backendGet: (o: unknown) => GPUTexture | undefined;
  res: number;
  sceneId: string;
  title: string;
  sppLadder?: number[]; // default [1, 2, 4, 8, 16]
  referenceSpp?: number; // default 256
  log?: (m: string) => void;
}

export interface GalleryCaptureResult {
  id: string;
  title: string;
  width: number;
  height: number;
  spp: number[];
  referenceSpp: number;
  // data URLs (image/png)
  color: Record<string, string>;
  reference: string;
  albedo: string;
  normal: string;
  // per-image sanity stats (computed from the exact bytes written into each PNG)
  stats: Record<string, ImageStats>;
}

interface ImageStats {
  width: number;
  height: number;
  meanLuma: number; // 0..255
  minLuma: number;
  maxLuma: number;
  nonBlackFraction: number; // fraction of pixels with luma > 4
  localVariance: number; // mean 3x3 luma variance (noise metric; higher = noisier)
}

function half2float(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function bytesToDataUrl(bytes: Uint8ClampedArray, w: number, h: number): string {
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  cv.getContext('2d')!.putImageData(new ImageData(bytes, w, h), 0, 0);
  return cv.toDataURL('image/png');
}

function computeStats(bytes: Uint8ClampedArray, w: number, h: number): ImageStats {
  let sum = 0;
  let min = 255;
  let max = 0;
  let nonBlack = 0;
  const luma = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const l = 0.299 * bytes[i * 4] + 0.587 * bytes[i * 4 + 1] + 0.114 * bytes[i * 4 + 2];
    luma[i] = l;
    sum += l;
    if (l < min) min = l;
    if (l > max) max = l;
    if (l > 4) nonBlack++;
  }
  // mean 3x3 local variance of luma (speckle / noise -> high)
  let vacc = 0;
  let vn = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let s = 0;
      let s2 = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const l = luma[(y + dy) * w + (x + dx)];
          s += l;
          s2 += l * l;
        }
      }
      const m = s / 9;
      vacc += s2 / 9 - m * m;
      vn++;
    }
  }
  return {
    width: w,
    height: h,
    meanLuma: sum / (w * h),
    minLuma: min,
    maxLuma: max,
    nonBlackFraction: nonBlack / (w * h),
    localVariance: vacc / vn,
  };
}

export function installGalleryCapture(deps: GalleryCaptureDeps): void {
  const {
    THREE, tsl, renderer, device, pathTracer, scene, camera,
    getTracerTexture, backendGet, res, sceneId, title,
  } = deps;
  const RES = res;
  const sppLadder = deps.sppLadder ?? [1, 2, 4, 8, 16];
  const referenceSpp = deps.referenceSpp ?? 256;
  const log = deps.log ?? ((m: string) => console.log(m));

  // Copy any GPUTexture (must be COPY_SRC) into a CPU buffer. bytesPerRow padded
  // to 256 (WebGPU requirement); we trim the padding on the way out.
  async function readTexture(tex: GPUTexture, bytesPerPixel: number): Promise<ArrayBuffer> {
    const unpadded = RES * bytesPerPixel;
    const bytesPerRow = Math.ceil(unpadded / 256) * 256;
    const buf = device.createBuffer({
      size: bytesPerRow * RES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: tex },
      { buffer: buf, bytesPerRow },
      { width: RES, height: RES },
    );
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buf.getMappedRange());
    // repack to tightly-packed rows
    const out = new Uint8Array(unpadded * RES);
    for (let y = 0; y < RES; y++) {
      out.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + unpadded), y * unpadded);
    }
    buf.unmap();
    buf.destroy();
    return out.buffer;
  }

  // Noisy color: read the tracer's linear-HDR float target, ACES-tonemap + sRGB
  // encode (Narkowicz ACES ~ three's ACESFilmicToneMapping), flip bottom-up ->
  // top-down. This is the exact display transform the demo's CPU-path button uses.
  async function captureColorBytes(): Promise<Uint8ClampedArray> {
    const tex = getTracerTexture();
    if (!tex) throw new Error('capture: tracer texture unavailable');
    const is32 = tex.format.includes('32float');
    const bpp = is32 ? 16 : 8;
    const raw = await readTexture(tex, bpp);
    const w = RES;
    const h = RES;
    const src = is32 ? new Float32Array(raw) : new Uint16Array(raw);
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      const srcY = h - 1 - y; // tracer target is bottom-up
      for (let x = 0; x < w; x++) {
        const si = (srcY * w + x) * 4;
        const di = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) {
          let v = is32 ? src[si + c] : half2float(src[si + c]);
          if (!Number.isFinite(v)) v = 0;
          // Narkowicz ACES approximation
          v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
          v = Math.min(1, Math.max(0, v));
          v = v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055; // linear->sRGB
          rgba[di + c] = v * 255;
        }
        rgba[di + 3] = 255;
      }
    }
    return rgba;
  }

  // --- AOV rendering ---------------------------------------------------------
  // Render each AOV into an rgba8 RenderTarget through a fullscreen quad that bakes
  // the exact encoding we want, then read the bytes straight back (no half-float
  // decode, no sRGB surprises). The raster G-buffer is top-down already, so the
  // quad output is top-down — matches the color capture and the network's aux
  // contract (auxInputFlipY: false).
  const { mrt, diffuseColor, normalView, texture: textureNode, vec4 } = tsl;
  const albedoRT = new THREE.RenderTarget(RES, RES, { type: THREE.HalfFloatType });
  const normalRT = new THREE.RenderTarget(RES, RES, { type: THREE.HalfFloatType });
  albedoRT.texture.name = 'albedo';
  normalRT.texture.name = 'normal';
  const encodeRT = new THREE.RenderTarget(RES, RES); // rgba8unorm, no color conversion
  const quad = new THREE.QuadMesh(new THREE.NodeMaterial());

  function forceOpaque(): Array<() => void> {
    const restore: Array<() => void> = [];
    scene.traverse((c: any) => {
      const mats: any[] = c.isMesh ? (Array.isArray(c.material) ? c.material : [c.material]) : [];
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

  function renderAOVs() {
    // albedo: background ON — the env color IS the correct albedo for env pixels
    // (OIDN convention; clamped to [0,1] on encode). First-hit, opaque.
    let restore = forceOpaque();
    try {
      renderer.setMRT(mrt({ albedo: diffuseColor }));
      renderer.setRenderTarget(albedoRT);
      renderer.render(scene, camera);
    } finally {
      renderer.setRenderTarget(null);
      renderer.setMRT(null);
      restore.forEach((fn) => fn());
    }
    // normal: background OFF — a cleared target reads normal=0 (OIDN's env
    // convention), instead of the env quad's meaningless normalView gradient.
    restore = forceOpaque();
    const bg = scene.background;
    scene.background = null;
    try {
      renderer.setMRT(mrt({ normal: normalView }));
      renderer.setRenderTarget(normalRT);
      renderer.render(scene, camera);
    } finally {
      renderer.setRenderTarget(null);
      renderer.setMRT(null);
      scene.background = bg;
      restore.forEach((fn) => fn());
    }
  }

  async function encodeAOV(kind: 'albedo' | 'normal'): Promise<Uint8ClampedArray> {
    const srcTex = kind === 'albedo' ? albedoRT.texture : normalRT.texture;
    const t = textureNode(srcTex);
    // albedo: linear [0,1] passed through (clamped) -> raw bytes = the [0,1] the
    // network expects. normal: [-1,1] -> [0,1] (mul .5 add .5) -> raw bytes.
    const rgb = kind === 'albedo'
      ? t.rgb.clamp(0, 1)
      : t.rgb.clamp(-1, 1).mul(0.5).add(0.5);
    (quad.material as any).fragmentNode = vec4(rgb, 1);
    (quad.material as any).needsUpdate = true;
    try {
      renderer.setRenderTarget(encodeRT);
      quad.render(renderer);
    } finally {
      renderer.setRenderTarget(null);
    }
    const gpu = backendGet(encodeRT.texture);
    if (!gpu) throw new Error(`capture: ${kind} encode texture unavailable`);
    const raw = await readTexture(gpu, 4);
    return new Uint8ClampedArray(raw);
  }

  async function accumulateTo(target: number, maxMs = 120000) {
    // Guard on both a generous call count and a wall-clock budget: renderSample
    // can advance `samples` by <1 per call in some tracer configs, so a tight
    // call cap would stop short of the target (the budget is the real backstop).
    const t0 = performance.now();
    let guard = 0;
    const hardCap = target * 50 + 1000;
    while (Math.floor(pathTracer.samples ?? 0) < target && guard++ < hardCap) {
      pathTracer.renderSample();
      if (guard % 8 === 0) {
        await device.queue.onSubmittedWorkDone();
        if (performance.now() - t0 > maxMs) break;
      }
    }
    await device.queue.onSubmittedWorkDone();
  }

  (window as any).__captureGallery = async (opts?: { referenceSpp?: number }): Promise<GalleryCaptureResult> => {
    const refSpp = opts?.referenceSpp ?? referenceSpp;
    (window as any).__capturing = true; // pause the example's own rAF loop
    try {
      log(`[capture] ${sceneId}: starting spp ladder ${sppLadder.join('/')} + ref ${refSpp}`);
      pathTracer.reset();
      await new Promise(requestAnimationFrame);

      const color: Record<string, string> = {};
      const stats: Record<string, ImageStats> = {};

      for (const spp of sppLadder) {
        await accumulateTo(spp);
        const bytes = await captureColorBytes();
        color[String(spp)] = bytesToDataUrl(bytes, RES, RES);
        stats[`color${spp}`] = computeStats(bytes, RES, RES);
        log(`[capture] ${sceneId}: spp${spp} at ${Math.floor(pathTracer.samples)} samples, var=${stats[`color${spp}`].localVariance.toFixed(1)}`);
      }

      await accumulateTo(refSpp);
      const refBytes = await captureColorBytes();
      const reference = bytesToDataUrl(refBytes, RES, RES);
      stats.reference = computeStats(refBytes, RES, RES);
      log(`[capture] ${sceneId}: reference at ${Math.floor(pathTracer.samples)} samples, var=${stats.reference.localVariance.toFixed(1)}`);

      renderAOVs();
      const albedoBytes = await encodeAOV('albedo');
      const normalBytes = await encodeAOV('normal');
      const albedo = bytesToDataUrl(albedoBytes, RES, RES);
      const normal = bytesToDataUrl(normalBytes, RES, RES);
      stats.albedo = computeStats(albedoBytes, RES, RES);
      stats.normal = computeStats(normalBytes, RES, RES);
      log(`[capture] ${sceneId}: AOVs done (albedo meanLuma=${stats.albedo.meanLuma.toFixed(1)}, normal var=${stats.normal.localVariance.toFixed(1)})`);

      const result: GalleryCaptureResult = {
        id: sceneId,
        title,
        width: RES,
        height: RES,
        spp: sppLadder.slice(),
        referenceSpp: Math.floor(pathTracer.samples),
        color,
        reference,
        albedo,
        normal,
        stats,
      };
      (window as any).__captureResult = result;
      log(`[capture] ${sceneId}: DONE`);
      return result;
    } finally {
      (window as any).__capturing = false;
    }
  };

  // Headed fallback: ?capture=1 mounts a button that runs the capture and
  // downloads the PNGs (+ a manifest fragment) so assets can be produced in a
  // real browser if headless init won't cooperate.
  const params = new URLSearchParams(location.search);
  if (params.has('capture')) {
    const bar = document.createElement('div');
    bar.style.cssText = 'position:fixed;top:8px;left:8px;z-index:99999;display:flex;gap:8px;align-items:center;background:rgba(11,14,20,0.85);padding:8px 10px;border-radius:8px;font:13px system-ui;color:#e2e8f0';
    const btn = document.createElement('button');
    btn.textContent = 'capture gallery assets';
    btn.style.cssText = 'padding:6px 12px;cursor:pointer';
    const msg = document.createElement('span');
    msg.textContent = `→ ${sceneId}`;
    bar.append(btn, msg);
    document.body.appendChild(bar);

    const download = (dataUrl: string, filename: string) => {
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };

    btn.addEventListener('click', async () => {
      btn.disabled = true;
      msg.textContent = 'capturing…';
      try {
        const r = await (window as any).__captureGallery();
        for (const spp of r.spp) download(r.color[String(spp)], `spp${spp}.png`);
        download(r.reference, 'reference.png');
        download(r.albedo, 'albedo.png');
        download(r.normal, 'normal.png');
        const manifestEntry = {
          id: r.id, title: r.title, width: r.width, height: r.height, spp: r.spp,
          color: Object.fromEntries(r.spp.map((s: number) => [String(s), `spp${s}.png`])),
          albedo: 'albedo.png', normal: 'normal.png', reference: 'reference.png',
        };
        download(
          'data:application/json,' + encodeURIComponent(JSON.stringify(manifestEntry, null, 2)),
          `${r.id}.manifest.json`,
        );
        msg.textContent = `done → save into public/scenes/${r.id}/`;
      } catch (e) {
        msg.textContent = 'ERROR: ' + (e as Error).message;
        console.error(e);
      } finally {
        btn.disabled = false;
      }
    });
  }
}
