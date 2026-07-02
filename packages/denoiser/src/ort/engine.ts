// DenoiseEngine: the WebGPU/ONNX-Runtime-Web inference core that replaces the old
// TensorFlow.js UNet + GPUTensorTiler. It owns the InferenceSession and the shared
// GPUDevice (ORT creates it; expose `device` to share with three.js — see #26107),
// does pre/post + tiling in WGSL (GpuImageOps), and keeps everything on the GPU
// except the final pixel readback.
//
// The ONNX models export with named free dims [batch, C, height, width]; the
// engine pins them per session via freeDimensionOverrides and keeps a small
// cache of sessions keyed by geometry. Per image it picks a plan:
//   - WHOLE-FRAME (preferred): pad W/H up to /16, batch=1, overlap=0 — one
//     session.run for the entire image, no overlap redundancy, no seams.
//   - TILED fallback (huge images / tight device limits): square tiles with
//     32px overlap, several tiles batched per run.
// The U-Net's full-res intermediates are large (up to ~96 channels × H × W), so
// whole-frame mode needs raised device limits — ORT requests a minimal device,
// so the FIRST session creation runs under a scoped requestAdapter patch that
// asks for the adapter's max limits/features (same trick as the pathtracer
// example, but restored immediately after).
import * as ort from 'onnxruntime-web/webgpu';
import { GpuImageOps } from './wgsl';

export interface EngineOptions {
  channels: number; // 3 | 6 | 9 — must match the model
  tile?: number; // base square tile for the tiled fallback (default 256)
  overlap?: number; // tile overlap in tiled mode (default 32)
  batch?: number; // max tiles per session.run in tiled mode (default 8)
  /** Model IO element type — must match the loaded .onnx (fp16 needs shader-f16). */
  precision?: 'fp32' | 'fp16';
  /**
   * Per-run pixel budget (tileW*tileH*batch). Images whose padded area fits run
   * whole-frame in one go; bigger images fall back to tiles. Default 2 359 296
   * (2048×1152 ≈ just over 1080p). Raise it on beefy GPUs for whole-frame 1440p+.
   */
  maxRunPixels?: number;
  wasmPaths?: string;
  /**
   * Opt-in WebGPU graph capture. Measured ~0 gain here (the tile loop is
   * GPU-bound, not dispatch-bound) and onnxruntime-web 1.27.0 crashes inside
   * its capture buffer manager at larger tile counts (createBindGroup:
   * "Required member is undefined", reproduced at 1080p/45 tiles while
   * 720p/24 tiles works). Off by default until upstream stabilizes.
   */
  graphCapture?: boolean;
}

export interface DenoiseOptions {
  albedo?: Uint8ClampedArray; // required when channels >= 6
  normal?: Uint8ClampedArray; // required when channels >= 9
  srgb?: boolean; // input is sRGB -> convert to linear before the model, back after
  hdr?: boolean; // skip the [0,1] output clamp
  flipY?: boolean; // flip the output vertically (in the resolve kernel, free)
  onProgress?: (p: number) => void;
}

/** Wall-clock stage timings for the last denoise() call (all in ms). */
export interface DenoiseStats {
  width: number;
  height: number;
  tiles: number;
  batches: number;
  tileW: number;
  tileH: number;
  batchSize: number;
  uploadMs: number; // input writeBuffer + accum/weight clear
  encodeMs: number; // WGSL extract/accumulate encode+submit (CPU side)
  runMs: number; // sum of awaited session.run() calls
  resolveMs: number; // resolve pass + readback map
  totalMs: number;
}

interface TileSpec {
  startX: number; startY: number; curW: number; curH: number; tx: number; ty: number;
}

interface Plan { tileW: number; tileH: number; batch: number; overlap: number; }

interface GeoSession {
  key: string;
  session: ort.InferenceSession;
  nchwInput: GPUBuffer;
  outNCHW: GPUBuffer;
  inputTensor: ort.Tensor;
  outputTensor: ort.Tensor;
  plan: Plan;
}

// Conservative upper bound on the widest full-resolution tensor in any of the
// U-Net variants (decoder concat levels), in channels. Used to test a candidate
// geometry against the device's buffer limits before creating a session.
const WORST_FULLRES_CHANNELS = 96;

const pad16 = (x: number) => Math.ceil(x / 16) * 16;

export class DenoiseEngine {
  device!: GPUDevice;
  inputName!: string;
  outputName!: string;
  channels = 3;
  readonly tile: number;
  readonly overlap: number;
  /** Max tiles per session.run in tiled mode. */
  batch: number;
  readonly precision: 'fp32' | 'fp16';
  maxRunPixels: number;
  /** Stage timings from the most recent denoise() call. */
  lastStats?: DenoiseStats;
  /** True when sessions run with WebGPU graph capture enabled. */
  graphCaptured = false;

  private modelBytes!: Uint8Array;
  private baseSessionOpts!: ort.InferenceSession.SessionOptions;
  private ops!: GpuImageOps;
  private geos = new Map<string, GeoSession>();
  private dynamicDims = true; // false for legacy static-dim models

  // per-image buffers
  private imgW = 0;
  private imgH = 0;
  private color?: GPUBuffer;
  private albedo?: GPUBuffer;
  private normal?: GPUBuffer;
  private accum?: GPUBuffer;
  private weight?: GPUBuffer;
  private outPixels?: GPUBuffer;
  private readback?: GPUBuffer;

  private constructor(opts: EngineOptions) {
    this.channels = opts.channels;
    this.tile = opts.tile ?? 256;
    this.overlap = opts.overlap ?? 32;
    this.batch = Math.max(1, opts.batch ?? 8);
    this.precision = opts.precision ?? 'fp32';
    this.maxRunPixels = opts.maxRunPixels ?? 2048 * 1152;
  }

  private get bpe() { return this.precision === 'fp16' ? 2 : 4; }

  static async create(modelBytes: Uint8Array, opts: EngineOptions): Promise<DenoiseEngine> {
    const e = new DenoiseEngine(opts);
    e.modelBytes = modelBytes;
    ort.env.wasm.numThreads = 1; // avoids needing cross-origin isolation
    ort.env.wasm.wasmPaths =
      opts.wasmPaths ?? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

    e.baseSessionOpts = {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
      graphOptimizationLevel: 'all',
    };
    if (opts.graphCapture) e.baseSessionOpts.enableGraphCapture = true;
    e.graphCaptured = !!opts.graphCapture;

    // Create the default (tiled-fallback) geometry now — this is also what
    // makes ORT create the GPUDevice, under the scoped max-limits patch.
    const deviceMissing = !(ort.env.webgpu as unknown as { device?: GPUDevice }).device;
    const unpatch = deviceMissing ? patchForMaxLimits() : undefined;
    try {
      await e.ensureGeo({ tileW: e.tile, tileH: e.tile, batch: e.batch, overlap: e.overlap });
    } finally {
      unpatch?.();
    }

    e.device = (ort.env.webgpu as unknown as { device: GPUDevice }).device;
    if (!e.device) throw new Error('Denoiser: ORT did not expose a WebGPU device');
    if (e.precision === 'fp16' && !e.device.features.has('shader-f16')) {
      // ORT requests shader-f16 on its device when the adapter has it; without
      // it our WGSL can't read/write the fp16 model IO buffers.
      e.dispose();
      throw new Error('Denoiser: fp16 needs the shader-f16 WebGPU feature (unavailable on this device)');
    }
    e.ops = new GpuImageOps(e.device, e.batch, e.precision === 'fp16');
    return e;
  }

  /** Get (or create) the session + IO buffers for a geometry. */
  private async ensureGeo(plan: Plan): Promise<GeoSession> {
    const key = `${plan.batch}|${plan.tileW}x${plan.tileH}`;
    const hit = this.geos.get(key);
    if (hit) return hit;

    let session: ort.InferenceSession;
    if (this.dynamicDims) {
      try {
        session = await ort.InferenceSession.create(this.modelBytes, {
          ...this.baseSessionOpts,
          freeDimensionOverrides: { batch: plan.batch, height: plan.tileH, width: plan.tileW },
        });
      } catch (err) {
        if (this.geos.size > 0) throw err; // dynamic already proven -> real failure
        // Legacy static-dim model ([1, C, 256, 256]): no free dims to pin.
        console.warn('Denoiser: model has static dims, geometry planning disabled', err);
        this.dynamicDims = false;
        this.batch = 1;
        plan = { tileW: 256, tileH: 256, batch: 1, overlap: this.overlap };
        session = await ort.InferenceSession.create(this.modelBytes, this.baseSessionOpts);
      }
    } else {
      session = await ort.InferenceSession.create(this.modelBytes, this.baseSessionOpts);
    }

    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
    const device = (ort.env.webgpu as unknown as { device: GPUDevice }).device;
    const els = plan.batch * plan.tileW * plan.tileH;
    const f16 = this.precision === 'fp16';
    const nchwInput = device.createBuffer({
      size: els * this.channels * this.bpe,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const outNCHW = device.createBuffer({
      size: els * 3 * this.bpe,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const geo: GeoSession = {
      key, session, nchwInput, outNCHW, plan,
      inputTensor: ort.Tensor.fromGpuBuffer(nchwInput, {
        dataType: f16 ? 'float16' : 'float32',
        dims: [plan.batch, this.channels, plan.tileH, plan.tileW],
      }),
      outputTensor: ort.Tensor.fromGpuBuffer(outNCHW, {
        dataType: f16 ? 'float16' : 'float32',
        dims: [plan.batch, 3, plan.tileH, plan.tileW],
      }),
    };
    // Tiny LRU: sessions hold a GPU copy of the weights; keep a few geometries.
    if (this.geos.size >= 4) {
      const oldest = this.geos.keys().next().value as string;
      this.releaseGeo(this.geos.get(oldest)!);
      this.geos.delete(oldest);
    }
    this.geos.set(key, geo);
    return geo;
  }

  private releaseGeo(g: GeoSession) {
    g.nchwInput.destroy();
    g.outNCHW.destroy();
    g.session.release?.();
  }

  /** Choose the run geometry for an image size (whole-frame when it fits). */
  planFor(w: number, h: number): Plan {
    if (!this.dynamicDims) return { tileW: 256, tileH: 256, batch: 1, overlap: this.overlap };
    const limits = this.device?.limits;
    const bufferCap = limits
      ? Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize)
      : 128 * 1024 * 1024;
    const worstBytes = (px: number) => px * WORST_FULLRES_CHANNELS * this.bpe;

    const pw = pad16(w);
    const ph = pad16(h);
    if (pw * ph <= this.maxRunPixels && worstBytes(pw * ph) <= bufferCap) {
      return { tileW: pw, tileH: ph, batch: 1, overlap: 0 };
    }
    for (const t of [1024, 512, this.tile]) {
      if (t > Math.max(pw, ph)) continue; // pointless: bigger than the image
      const perTile = t * t;
      const batch = Math.min(
        this.batch,
        Math.max(1, Math.floor(this.maxRunPixels / perTile)),
        Math.max(1, Math.floor(bufferCap / worstBytes(perTile))),
      );
      if (worstBytes(perTile * batch) <= bufferCap || batch === 1) {
        if (worstBytes(perTile) <= bufferCap) return { tileW: t, tileH: t, batch, overlap: this.overlap };
      }
    }
    return { tileW: this.tile, tileH: this.tile, batch: 1, overlap: this.overlap };
  }

  private ensureImageBuffers(w: number, h: number) {
    if (this.imgW === w && this.imgH === h && this.color) return;
    [this.color, this.albedo, this.normal, this.accum, this.weight, this.outPixels, this.readback]
      .forEach((b) => b?.destroy());
    const d = this.device;
    const px = w * h;
    const stor = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.color = d.createBuffer({ size: px * 4, usage: stor });
    this.albedo = this.channels >= 6 ? d.createBuffer({ size: px * 4, usage: stor }) : undefined;
    this.normal = this.channels >= 9 ? d.createBuffer({ size: px * 4, usage: stor }) : undefined;
    this.accum = d.createBuffer({ size: 3 * px * 4, usage: stor });
    this.weight = d.createBuffer({ size: px * 4, usage: stor });
    this.outPixels = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readback = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.imgW = w;
    this.imgH = h;
  }

  /** Denoise a full-resolution image (whole-frame or tiled+blended). Returns RGBA8 pixels (alpha = 255). */
  async denoise(color: Uint8ClampedArray, w: number, h: number, opts: DenoiseOptions = {}): Promise<Uint8ClampedArray> {
    if (color.length !== w * h * 4) throw new Error(`Denoiser: expected ${w * h * 4} color bytes, got ${color.length}`);
    if (this.channels >= 6 && !opts.albedo) throw new Error('Denoiser: model requires an albedo input');
    if (this.channels >= 9 && !opts.normal) throw new Error('Denoiser: model requires a normal input');

    const geo = await this.ensureGeo(this.planFor(w, h));
    const { tileW, tileH, batch: B, overlap } = geo.plan;

    this.ensureImageBuffers(w, h);
    const d = this.device;
    const strideX = tileW - overlap;
    const strideY = tileH - overlap;
    const tilesX = Math.max(1, Math.ceil((w - overlap) / strideX));
    const tilesY = Math.max(1, Math.ceil((h - overlap) / strideY));

    const tiles: TileSpec[] = [];
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const startX = tx * strideX;
        const startY = ty * strideY;
        tiles.push({
          startX, startY, tx, ty,
          curW: Math.min(tileW, w - startX),
          curH: Math.min(tileH, h - startY),
        });
      }
    }
    const total = tiles.length;
    const batches = Math.ceil(total / B);

    const tStart = performance.now();
    d.queue.writeBuffer(this.color!, 0, color);
    if (opts.albedo) d.queue.writeBuffer(this.albedo!, 0, opts.albedo);
    if (opts.normal) d.queue.writeBuffer(this.normal!, 0, opts.normal);

    const clr = d.createCommandEncoder();
    clr.clearBuffer(this.accum!);
    clr.clearBuffer(this.weight!);
    d.queue.submit([clr.finish()]);
    const tUpload = performance.now();

    const albedoBuf = this.albedo ?? this.color!;
    const normalBuf = this.normal ?? this.color!;
    const offsets = new Uint32Array(Math.max(this.ops.maxBatch, B) * 2);
    let encodeMs = 0;
    let runMs = 0;
    let done = 0;
    for (let b0 = 0; b0 < total; b0 += B) {
      const chunk = tiles.slice(b0, b0 + B);

      let t0 = performance.now();
      for (let i = 0; i < chunk.length; i++) {
        offsets[i * 2] = chunk[i].startX;
        offsets[i * 2 + 1] = chunk[i].startY;
      }
      const e1 = d.createCommandEncoder();
      this.ops.encodeExtractTiles(
        e1, this.color!, albedoBuf, normalBuf, geo.nchwInput,
        w, h, tileW, tileH, this.channels, !!opts.srgb, offsets, chunk.length);
      d.queue.submit([e1.finish()]);
      const t1 = performance.now();
      encodeMs += t1 - t0;

      // Unused slots of a short final batch still run through the model with
      // stale (valid float) contents; their outputs are simply never blended.
      await geo.session.run(
        { [this.inputName]: geo.inputTensor },
        { [this.outputName]: geo.outputTensor });
      t0 = performance.now();
      runMs += t0 - t1;

      const e2 = d.createCommandEncoder();
      chunk.forEach((tl, i) => {
        this.ops.encodeAccumulateTile(e2, i, geo.outNCHW, this.accum!, this.weight!, {
          imgW: w, imgH: h, startX: tl.startX, startY: tl.startY, curW: tl.curW, curH: tl.curH,
          tileX: tl.tx, tileY: tl.ty, tilesX, tilesY, tileW, tileH, overlap,
          batchIdx: i,
        });
      });
      d.queue.submit([e2.finish()]);
      encodeMs += performance.now() - t0;
      done += chunk.length;
      opts.onProgress?.(done / total);
    }

    const tTiles = performance.now();
    const e3 = d.createCommandEncoder();
    this.ops.encodeResolve(
      e3, this.accum!, this.weight!, this.outPixels!, w, h, !!opts.srgb, !!opts.hdr, !!opts.flipY);
    e3.copyBufferToBuffer(this.outPixels!, 0, this.readback!, 0, w * h * 4);
    d.queue.submit([e3.finish()]);

    await this.readback!.mapAsync(GPUMapMode.READ);
    const out = new Uint8ClampedArray(this.readback!.getMappedRange().slice(0));
    this.readback!.unmap();
    const tEnd = performance.now();
    this.lastStats = {
      width: w, height: h, tiles: total, batches,
      tileW, tileH, batchSize: B,
      uploadMs: tUpload - tStart,
      encodeMs, runMs,
      resolveMs: tEnd - tTiles,
      totalMs: tEnd - tStart,
    };
    return out;
  }

  tileGrid(w: number, h: number) {
    const plan = this.planFor(w, h);
    const strideX = plan.tileW - plan.overlap;
    const strideY = plan.tileH - plan.overlap;
    return {
      tilesX: Math.max(1, Math.ceil((w - plan.overlap) / strideX)),
      tilesY: Math.max(1, Math.ceil((h - plan.overlap) / strideY)),
    };
  }

  dispose() {
    [this.color, this.albedo, this.normal, this.accum, this.weight, this.outPixels, this.readback]
      .forEach((b) => b?.destroy());
    this.geos.forEach((g) => this.releaseGeo(g));
    this.geos.clear();
  }
}

/**
 * Temporarily patch requestAdapter so the device ORT is about to create gets
 * the adapter's FULL limits + features (ORT requests a minimal device, which
 * caps storage buffers at ~128-256MB — far too small for whole-frame U-Net
 * intermediates, and too small for three.js path tracers sharing the device).
 * Returns a restore function.
 */
function patchForMaxLimits(): () => void {
  if (!('gpu' in navigator)) return () => undefined;
  const gpu = navigator.gpu as GPU;
  const origRequestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (adapterOpts?: GPURequestAdapterOptions) => {
    const adapter = await origRequestAdapter(adapterOpts);
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
  return () => { gpu.requestAdapter = origRequestAdapter; };
}
