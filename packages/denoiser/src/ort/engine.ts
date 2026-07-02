// DenoiseEngine: the WebGPU/ONNX-Runtime-Web inference core that replaces the old
// TensorFlow.js UNet + GPUTensorTiler. It owns the InferenceSession and the shared
// GPUDevice (ORT creates it; expose `device` to share with three.js — see #26107),
// does pre/post + tiling in WGSL (GpuImageOps), and keeps everything on the GPU
// except the final pixel readback.
//
// Ported and generalized from examples/webgpu-ort-smoke (browser-verified).
import * as ort from 'onnxruntime-web/webgpu';
import { GpuImageOps } from './wgsl';

export interface EngineOptions {
  channels: number; // 3 | 6 | 9 — must match the model
  tile?: number; // model's fixed square size (default 256)
  overlap?: number; // tile overlap (default 32)
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
  onProgress?: (p: number) => void;
}

/** Wall-clock stage timings for the last denoise() call (all in ms). */
export interface DenoiseStats {
  width: number;
  height: number;
  tiles: number;
  uploadMs: number; // input writeBuffer + accum/weight clear
  encodeMs: number; // WGSL extract/accumulate encode+submit (CPU side)
  runMs: number; // sum of awaited session.run() calls
  resolveMs: number; // resolve pass + readback map
  totalMs: number;
}

export class DenoiseEngine {
  private session!: ort.InferenceSession;
  device!: GPUDevice;
  inputName!: string;
  outputName!: string;
  channels = 3;
  readonly tile: number;
  readonly overlap: number;
  /** Stage timings from the most recent denoise() call. */
  lastStats?: DenoiseStats;
  /** True when the session runs with WebGPU graph capture enabled. */
  graphCaptured = false;

  private ops!: GpuImageOps;
  private nchwInput!: GPUBuffer;
  private outNCHW!: GPUBuffer;
  private inputTensor!: ort.Tensor;
  private outputTensor!: ort.Tensor;

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
  }

  static async create(modelBytes: Uint8Array, opts: EngineOptions): Promise<DenoiseEngine> {
    const e = new DenoiseEngine(opts);
    ort.env.wasm.numThreads = 1; // avoids needing cross-origin isolation
    ort.env.wasm.wasmPaths =
      opts.wasmPaths ?? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

    const sessionOpts: ort.InferenceSession.SessionOptions = {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
      graphOptimizationLevel: 'all',
    };
    // Graph capture records the U-Net's GPU commands once and replays them on
    // later runs, skipping per-run graph walking/dispatch. It requires static
    // shapes + every op on the WebGPU EP + gpu-buffer IO (all true here), and
    // session creation throws if the model doesn't qualify — fall back cleanly.
    // NOTE: capture binds IO buffers on the FIRST run and never re-binds; the
    // engine must keep reusing (and mutating) the same input/output GPUBuffers.
    // Opt-in only — see EngineOptions.graphCapture for the ORT 1.27 crash.
    if (opts.graphCapture) {
      try {
        e.session = await ort.InferenceSession.create(modelBytes, {
          ...sessionOpts,
          enableGraphCapture: true,
        });
        e.graphCaptured = true;
      } catch (err) {
        console.warn('Denoiser: graph capture unavailable, falling back', err);
      }
    }
    if (!e.session) e.session = await ort.InferenceSession.create(modelBytes, sessionOpts);
    e.inputName = e.session.inputNames[0];
    e.outputName = e.session.outputNames[0];
    e.device = (ort.env.webgpu as unknown as { device: GPUDevice }).device;
    if (!e.device) throw new Error('Denoiser: ORT did not expose a WebGPU device');

    e.ops = new GpuImageOps(e.device);
    const t = e.tile;
    e.nchwInput = e.device.createBuffer({
      size: e.channels * t * t * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    e.outNCHW = e.device.createBuffer({
      size: 3 * t * t * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    e.inputTensor = ort.Tensor.fromGpuBuffer(e.nchwInput, {
      dataType: 'float32', dims: [1, e.channels, t, t],
    });
    e.outputTensor = ort.Tensor.fromGpuBuffer(e.outNCHW, {
      dataType: 'float32', dims: [1, 3, t, t],
    });
    return e;
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

  /** Denoise a full-resolution image (tiled, blended). Returns RGBA8 pixels (alpha = 255). */
  async denoise(color: Uint8ClampedArray, w: number, h: number, opts: DenoiseOptions = {}): Promise<Uint8ClampedArray> {
    if (color.length !== w * h * 4) throw new Error(`Denoiser: expected ${w * h * 4} color bytes, got ${color.length}`);
    if (this.channels >= 6 && !opts.albedo) throw new Error('Denoiser: model requires an albedo input');
    if (this.channels >= 9 && !opts.normal) throw new Error('Denoiser: model requires a normal input');

    this.ensureImageBuffers(w, h);
    const d = this.device;
    const TILE = this.tile;
    const stride = TILE - this.overlap;
    const tilesX = Math.ceil(w / stride);
    const tilesY = Math.ceil(h / stride);
    const total = tilesX * tilesY;

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
    let encodeMs = 0;
    let runMs = 0;
    let done = 0;
    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const startX = tx * stride;
        const startY = ty * stride;
        const curW = Math.min(TILE, w - startX);
        const curH = Math.min(TILE, h - startY);

        let t0 = performance.now();
        const e1 = d.createCommandEncoder();
        this.ops.encodeExtractTile(
          e1, this.color!, albedoBuf, normalBuf, this.nchwInput!,
          w, h, startX, startY, TILE, this.channels, !!opts.srgb);
        d.queue.submit([e1.finish()]);
        let t1 = performance.now();
        encodeMs += t1 - t0;

        await this.session.run(
          { [this.inputName]: this.inputTensor },
          { [this.outputName]: this.outputTensor });
        t0 = performance.now();
        runMs += t0 - t1;

        const e2 = d.createCommandEncoder();
        this.ops.encodeAccumulateTile(e2, this.outNCHW!, this.accum!, this.weight!, {
          imgW: w, imgH: h, startX, startY, curW, curH,
          tileX: tx, tileY: ty, tilesX, tilesY, tile: TILE, overlap: this.overlap,
        });
        d.queue.submit([e2.finish()]);
        encodeMs += performance.now() - t0;
        opts.onProgress?.(++done / total);
      }
    }

    const tTiles = performance.now();
    const e3 = d.createCommandEncoder();
    this.ops.encodeResolve(e3, this.accum!, this.weight!, this.outPixels!, w, h, !!opts.srgb, !!opts.hdr);
    e3.copyBufferToBuffer(this.outPixels!, 0, this.readback!, 0, w * h * 4);
    d.queue.submit([e3.finish()]);

    await this.readback!.mapAsync(GPUMapMode.READ);
    const out = new Uint8ClampedArray(this.readback!.getMappedRange().slice(0));
    this.readback!.unmap();
    const tEnd = performance.now();
    this.lastStats = {
      width: w, height: h, tiles: total,
      uploadMs: tUpload - tStart,
      encodeMs, runMs,
      resolveMs: tEnd - tTiles,
      totalMs: tEnd - tStart,
    };
    return out;
  }

  tileGrid(w: number, h: number) {
    const stride = this.tile - this.overlap;
    return { tilesX: Math.ceil(w / stride), tilesY: Math.ceil(h / stride) };
  }

  dispose() {
    [this.color, this.albedo, this.normal, this.accum, this.weight, this.outPixels, this.readback,
      this.nchwInput, this.outNCHW].forEach((b) => b?.destroy());
    this.session?.release?.();
  }
}
