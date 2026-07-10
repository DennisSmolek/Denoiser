import { Denoiser } from 'denoiser';
import { ensureWebGPU, demoFooter, statsOverlay } from '../../_shared/chrome';
import pathTracerWGSL from './pathtracer.wgsl?raw';
import presentWGSL from './present.wgsl?raw';

// A plain-WebGPU toy path tracer (no three.js, no libraries) that shares ONE
// GPUDevice with the denoiser. The scene accumulates noisy path-traced samples
// into a linear-HDR texture; toggling "Live denoise" hands that texture straight
// to denoiser.denoiseTextures() every N samples. See the "denoiser integration"
// block below — that ~15-line recipe is the whole point of this demo.

const W = 512;
const H = 512;
const MAX_SAMPLES = 2000; // stop tracing once effectively converged

// ---- DOM --------------------------------------------------------------------
const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!;
const denoiseToggle = document.querySelector<HTMLInputElement>('#denoise-toggle')!;
const auxToggle = document.querySelector<HTMLInputElement>('#aux-toggle')!;
const intervalControl = document.querySelector<HTMLElement>('#interval-control')!;
const sppBadge = document.querySelector<HTMLElement>('#spp-badge')!;
const viewBadge = document.querySelector<HTMLElement>('#view-badge')!;
const statusEl = document.querySelector<HTMLElement>('#status')!;
const loadingEl = document.querySelector<HTMLElement>('#loading')!;

// ---- state ------------------------------------------------------------------
let liveDenoise = false;
let auxOn = true;
let interval = 16; // denoise every N samples
let samples = 0;
let denoising = false;

async function main(): Promise<void> {
  if (!(await ensureWebGPU())) return;

  loadingEl.textContent = 'fetching model + creating WebGPU device...';

  // ===========================================================================
  // === denoiser integration — the whole recipe (copy this) ===================
  // ===========================================================================
  // 1. Create the denoiser FIRST. onnxruntime-web OWNS GPUDevice creation (it
  //    ignores an injected one), so it must make the device before you build any
  //    of your own pipelines. Then share `denoiser.device` — do NOT call
  //    navigator.gpu.requestAdapter()/requestDevice() yourself.
  const denoiser = await Denoiser.create({ precision: 'fp16' });
  const device = denoiser.device;

  // 2. Build ALL your own WebGPU pipelines on that shared device (see below).

  // 3. Every N samples (and once converged), hand your linear-HDR accumulation
  //    texture straight to the denoiser. `hdr:true` applies OIDN's PU transfer +
  //    autoexposure; `transfer:'aces-srgb'` returns a display-ready rgba8unorm
  //    texture you can blit to the canvas. Aux inputs are optional GPUTextures.
  async function denoiseFrame(): Promise<GPUTexture | undefined> {
    return denoiser.denoiseTextures({
      color: accum[frontIdx],                    // rgba16float, linear-HDR radiance (running mean)
      // Omit the aux keys entirely when unused — passing `albedo: undefined`
      // reads as "I gave you a texture and it failed to unwrap" and warns.
      ...(auxOn ? { albedo: albedoTex, normal: normalTex } : {}), // [0,1] linear / [-1,1] floats (env=0)
      hdr: true,                                 // OIDN PU transfer + autoexposure
      transfer: 'aces-srgb',                     // ACES + sRGB -> display-ready rgba8unorm
    });
  }
  // ===========================================================================

  loadingEl.textContent = 'building path-tracer pipelines...';

  // Ping-ponged linear-HDR accumulation textures (running mean). Each frame the
  // compute pass reads last frame's mean and writes the new one; both need
  // STORAGE_BINDING (write) + TEXTURE_BINDING (read by denoiser/present).
  const accumUsage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const accum: [GPUTexture, GPUTexture] = [
    device.createTexture({ size: [W, H], format: 'rgba16float', usage: accumUsage }),
    device.createTexture({ size: [W, H], format: 'rgba16float', usage: accumUsage }),
  ];
  // Aux planes (first-hit albedo/normal), rewritten each frame by the same kernel.
  const albedoTex = device.createTexture({ size: [W, H], format: 'rgba16float', usage: accumUsage });
  const normalTex = device.createTexture({ size: [W, H], format: 'rgba16float', usage: accumUsage });

  const uniformBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // Compute pipeline (the path tracer).
  const traceModule = device.createShaderModule({ code: pathTracerWGSL });
  const tracePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: traceModule, entryPoint: 'main' } });
  // Two bind groups: index i READS accum[i] and WRITES accum[1-i].
  const traceBind = [0, 1].map((i) =>
    device.createBindGroup({
      layout: tracePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: accum[i].createView() },
        { binding: 1, resource: accum[1 - i].createView() },
        { binding: 2, resource: albedoTex.createView() },
        { binding: 3, resource: normalTex.createView() },
        { binding: 4, resource: { buffer: uniformBuf } },
      ],
    }),
  );

  // Present pipeline (tonemap blit for the RAW view) -> an offscreen rgba8unorm
  // texture we then copy to the canvas. Denoised frames skip this and copy the
  // denoiser's own output texture instead; both paths end with one copy.
  const presentModule = device.createShaderModule({ code: presentWGSL });
  const presentPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: presentModule, entryPoint: 'vs' },
    fragment: { module: presentModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  });
  const frameTex = device.createTexture({
    size: [W, H], format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const presentBind = [0, 1].map((i) =>
    device.createBindGroup({
      layout: presentPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: accum[i].createView() }],
    }),
  );

  // Canvas: rgba8unorm so we can copyTextureToTexture the display-ready result
  // (raw frameTex or denoiser output) straight into it.
  const ctx = canvas.getContext('webgpu')!;
  ctx.configure({ device, format: 'rgba8unorm', usage: GPUTextureUsage.COPY_DST });

  // Ping-pong bookkeeping: readIdx = texture the next compute reads; frontIdx =
  // texture holding the latest mean (for present/denoise).
  let readIdx = 0;
  let frontIdx = 1;

  const stats = statsOverlay();

  function traceOneSample(): void {
    device.queue.writeBuffer(uniformBuf, 0, new Uint32Array([W, H, samples, auxOn ? 1 : 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(tracePipeline);
    pass.setBindGroup(0, traceBind[readIdx]);
    pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8));
    pass.end();
    device.queue.submit([enc.finish()]);
    frontIdx = 1 - readIdx;
    readIdx = frontIdx;
    samples++;
  }

  function copyToCanvas(src: GPUTexture): void {
    const enc = device.createCommandEncoder();
    enc.copyTextureToTexture({ texture: src }, { texture: ctx.getCurrentTexture() }, { width: W, height: H });
    device.queue.submit([enc.finish()]);
  }

  function presentRaw(): void {
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: frameTex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(presentPipeline);
    pass.setBindGroup(0, presentBind[frontIdx]);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
    copyToCanvas(frameTex);
    viewBadge.textContent = 'raw';
  }

  async function presentDenoised(): Promise<void> {
    const t0 = performance.now();
    const out = await denoiseFrame();
    if (!out) return;
    copyToCanvas(out);
    stats.frame(performance.now() - t0);
    const s = denoiser.stats;
    const model = denoiser.modelName ?? '?';
    const ch = auxOn ? 9 : 3;
    const splitNote = auxOn ? ' (splitAux)' : '';
    statusEl.innerHTML =
      `model <span class="model">${model}</span> · ${ch}ch${splitNote} · ` +
      `denoised in ${s?.totalMs.toFixed(1) ?? '?'} ms · ${W}×${H} · ${samples} spp`;
    viewBadge.textContent = `denoised · ${s?.totalMs.toFixed(0) ?? '?'}ms`;
  }

  function updateBadges(): void {
    const conv = samples >= MAX_SAMPLES ? ' · converged' : '';
    sppBadge.textContent = `${samples} spp${conv}`;
  }

  // ---- render loop ----------------------------------------------------------
  function frame(): void {
    if (samples < MAX_SAMPLES) traceOneSample();
    updateBadges();
    if (liveDenoise) {
      if (!denoising && (samples % interval === 0 || samples >= MAX_SAMPLES)) {
        denoising = true;
        void presentDenoised().finally(() => { denoising = false; });
      }
    } else {
      presentRaw();
      if (!statusEl.textContent) {
        statusEl.innerHTML = `raw accumulation · ${W}×${H} · toggle <em>Live denoise</em> to clean it up`;
      }
    }
    requestAnimationFrame(frame);
  }

  // ---- controls -------------------------------------------------------------
  denoiseToggle.addEventListener('change', () => { liveDenoise = denoiseToggle.checked; statusEl.textContent = ''; });
  auxToggle.addEventListener('change', () => { auxOn = auxToggle.checked; });
  for (const n of [8, 16, 32, 64]) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = String(n);
    btn.setAttribute('aria-pressed', String(n === interval));
    btn.addEventListener('click', () => {
      interval = n;
      for (const b of intervalControl.children) b.setAttribute('aria-pressed', String(b === btn));
    });
    intervalControl.appendChild(btn);
  }

  loadingEl.textContent = '';
  requestAnimationFrame(frame);

  // ---- headless verification hooks (no-ops for real users) ------------------
  const readTex = async (tex: GPUTexture): Promise<Uint8Array> => {
    const bpr = W * 4; // 2048 — already a multiple of 256
    const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, { width: W, height: H });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    return out;
  };
  // Noise proxy: mean absolute luma difference between adjacent pixels (lower = smoother).
  const metric = (px: Uint8Array) => {
    const luma = (i: number) => 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
    let noise = 0, sum = 0, n = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const l = luma(i);
        sum += l; n++;
        if (x + 1 < W) noise += Math.abs(l - luma(i + 4));
        if (y + 1 < H) noise += Math.abs(l - luma(i + W * 4));
      }
    }
    return { meanLuma: +(sum / n).toFixed(2), noise: +(noise / (n * 2)).toFixed(3) };
  };
  const win = window as unknown as Record<string, unknown>;
  win.__spp = () => samples;
  win.__view = () => viewBadge.textContent;
  win.__setDenoise = (v: boolean) => { denoiseToggle.checked = v; denoiseToggle.dispatchEvent(new Event('change')); };
  win.__setAux = (v: boolean) => { auxToggle.checked = v; auxToggle.dispatchEvent(new Event('change')); };
  win.__measureRaw = async () => { presentRaw(); return { spp: samples, ...metric(await readTex(frameTex)) }; };
  win.__measureDenoised = async () => {
    const out = await denoiseFrame();
    if (!out) return null;
    return { spp: samples, stats: denoiser.stats, ...metric(await readTex(out)) };
  };
  statusEl.setAttribute('data-ready', '1');
  console.info('[webgpu-raw] path tracer initialized');
}

demoFooter('webgpu-raw');
main().catch((err) => {
  console.error(err);
  statusEl.textContent = `ERROR: ${(err as Error).message}`;
  loadingEl.textContent = '';
});
