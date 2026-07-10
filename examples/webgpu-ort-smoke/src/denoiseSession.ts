// Minimal ORT-Web WebGPU inference wrapper for a converted OIDN U-Net.
//
// This is the seed of the future library core (packages/denoiser/src/ort/*):
// it proves the WebGPU EP + zero-copy GPU-buffer IO path end to end. It runs a
// single fixed-size tile (no tiling/blending yet) on an already-normalized,
// NCHW float32 input — exactly the contract of the exported .onnx models.
import * as ort from 'onnxruntime-web/webgpu';
import { GpuImageOps } from './gpuOps';

export interface DenoiseSessionOptions {
  /** URL of the .onnx model (fixed-shape, NCHW). */
  modelUrl: string;
  /** Override where ORT loads its wasm/jsep assets from (default: jsDelivr CDN). */
  wasmPaths?: string;
  /** Input channels (3 color, 6 +albedo, 9 +albedo+normal). */
  channels?: number;
  /** Square tile size the model was exported with. */
  size?: number;
}

export class DenoiseSession {
  private session!: ort.InferenceSession;
  /** The GPUDevice ORT created — share this with three.js WebGPURenderer (issue #26107). */
  device!: GPUDevice;
  inputName!: string;
  outputName!: string;
  channels = 3;
  size = 256;

  // Full-GPU path resources (created lazily on first denoiseTileGPU call)
  private ops?: GpuImageOps;
  private inputPixels?: GPUBuffer; // RGBA8, size*size (color, 3ch path)
  private inputAlbedo?: GPUBuffer; // RGBA8, size*size (aux path)
  private inputNormal?: GPUBuffer; // RGBA8, size*size (aux path)
  private nchwInput?: GPUBuffer; // f32 NCHW, bound as ORT input
  private outNCHW?: GPUBuffer; // f32 NCHW, bound as ORT output
  private outPixels?: GPUBuffer; // RGBA8 result
  private readback?: GPUBuffer; // MAP_READ
  private inputTensor?: ort.Tensor;
  private outputTensor?: ort.Tensor;

  // Full-image (tiled) path resources, (re)allocated per image size
  private imgW = 0;
  private imgH = 0;
  private imgPixels?: GPUBuffer;
  private accum?: GPUBuffer;
  private weight?: GPUBuffer;
  private outPixelsImg?: GPUBuffer;
  private imgReadback?: GPUBuffer;

  readonly overlap = 32;

  async init(opts: DenoiseSessionOptions): Promise<void> {
    this.channels = opts.channels ?? 3;
    this.size = opts.size ?? 256;

    // Single-threaded avoids needing cross-origin isolation (COOP/COEP) headers.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths =
      opts.wasmPaths ?? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

    const bytes = new Uint8Array(await (await fetch(opts.modelUrl)).arrayBuffer());
    this.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['webgpu'],
      preferredOutputLocation: 'gpu-buffer',
      graphOptimizationLevel: 'all',
    });

    this.inputName = this.session.inputNames[0];
    this.outputName = this.session.outputNames[0];
    this.device = (ort.env.webgpu as unknown as { device: GPUDevice }).device;
    if (!this.device) throw new Error('ORT did not expose a WebGPU device');
  }

  /**
   * Run inference on a single tile.
   * @param nchw normalized input, layout [1, channels, size, size], float32
   * @returns denoised output, layout [1, 3, size, size], float32
   */
  async denoiseTile(nchw: Float32Array): Promise<Float32Array> {
    const expected = this.channels * this.size * this.size;
    if (nchw.length !== expected)
      throw new Error(`Expected ${expected} input elements, got ${nchw.length}`);

    // Upload input into a GPU storage buffer (size padded to a multiple of 16).
    const byteSize = nchw.byteLength;
    const inputBuf = this.device.createBuffer({
      size: Math.ceil(byteSize / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(inputBuf, 0, nchw.buffer, nchw.byteOffset, byteSize);

    const input = ort.Tensor.fromGpuBuffer(inputBuf, {
      dataType: 'float32',
      dims: [1, this.channels, this.size, this.size],
    });

    const results = await this.session.run({ [this.inputName]: input });
    const out = results[this.outputName];
    // download to CPU and release the GPU copy (true = release after read)
    const data = (await out.getData(true)) as Float32Array;

    // we own inputBuf (fromGpuBuffer doesn't take ownership) — destroy it ourselves
    inputBuf.destroy();
    return data;
  }

  private setupGpuPath() {
    const d = this.device;
    const px = this.size * this.size;
    this.ops = new GpuImageOps(d);
    this.inputPixels = d.createBuffer({
      size: px * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (this.channels >= 9) {
      const auxUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      this.inputAlbedo = d.createBuffer({ size: px * 4, usage: auxUsage });
      this.inputNormal = d.createBuffer({ size: px * 4, usage: auxUsage });
    }
    this.nchwInput = d.createBuffer({
      size: this.channels * px * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.outNCHW = d.createBuffer({
      size: 3 * px * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.outPixels = d.createBuffer({
      size: px * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.readback = d.createBuffer({
      size: px * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    // Persistent ORT IO tensors bound to the fixed buffers (reused every run).
    this.inputTensor = ort.Tensor.fromGpuBuffer(this.nchwInput, {
      dataType: 'float32',
      dims: [1, this.channels, this.size, this.size],
    });
    this.outputTensor = ort.Tensor.fromGpuBuffer(this.outNCHW, {
      dataType: 'float32',
      dims: [1, 3, this.size, this.size],
    });
  }

  /**
   * Full-GPU denoise of one tile: RGBA8 in -> RGBA8 out, with normalization,
   * layout conversion, inference and de-normalization all on the shared device.
   * Only the final pixels are read back (for a 2D canvas).
   */
  async denoiseTileGPU(rgba: Uint8ClampedArray): Promise<Uint8ClampedArray> {
    if (rgba.length !== this.size * this.size * 4)
      throw new Error(`Expected ${this.size * this.size * 4} RGBA bytes, got ${rgba.length}`);
    if (!this.ops) this.setupGpuPath();
    const d = this.device;

    // 1) upload pixels, 2) RGBA8 -> normalized NCHW
    // TS 5.7+ lib.dom typings narrowed GPUAllowSharedBufferSource views to ArrayBuffer-backed;
    // this example never uses SharedArrayBuffer, so this cast is safe.
    d.queue.writeBuffer(this.inputPixels!, 0, rgba as BufferSource);
    const pre = d.createCommandEncoder();
    this.ops!.encodeToNCHW(pre, this.inputPixels!, this.nchwInput!, this.size, this.size);
    d.queue.submit([pre.finish()]);

    // 3) inference, writing into our bound output buffer (no per-run alloc)
    await this.session.run(
      { [this.inputName]: this.inputTensor! },
      { [this.outputName]: this.outputTensor! },
    );

    // 4) NCHW -> RGBA8, copy to mappable buffer, 5) read back
    const post = d.createCommandEncoder();
    this.ops!.encodeToRGBA(post, this.outNCHW!, this.outPixels!, this.size, this.size);
    post.copyBufferToBuffer(this.outPixels!, 0, this.readback!, 0, this.size * this.size * 4);
    d.queue.submit([post.finish()]);

    await this.readback!.mapAsync(GPUMapMode.READ);
    const out = new Uint8ClampedArray(this.readback!.getMappedRange().slice(0));
    this.readback!.unmap();
    return out;
  }

  /**
   * Single-tile denoise with auxiliary inputs: color + albedo + normal RGBA8 ->
   * 9-channel NCHW (concat, color/albedo [0,1], normal [-1,1]) -> model -> RGBA8.
   * Requires a 9-channel model (e.g. rt_ldr_alb_nrm_small).
   */
  async denoiseAuxTile(
    color: Uint8ClampedArray, albedo: Uint8ClampedArray, normal: Uint8ClampedArray,
  ): Promise<Uint8ClampedArray> {
    if (this.channels < 9) throw new Error('denoiseAuxTile needs a 9-channel model');
    if (!this.ops) this.setupGpuPath();
    const d = this.device;

    // See ArrayBuffer-vs-ArrayBufferLike note above.
    d.queue.writeBuffer(this.inputPixels!, 0, color as BufferSource);
    d.queue.writeBuffer(this.inputAlbedo!, 0, albedo as BufferSource);
    d.queue.writeBuffer(this.inputNormal!, 0, normal as BufferSource);

    const pre = d.createCommandEncoder();
    this.ops!.encodeToNCHWAux(
      pre, this.inputPixels!, this.inputAlbedo!, this.inputNormal!, this.nchwInput!, this.size, this.size);
    d.queue.submit([pre.finish()]);

    await this.session.run(
      { [this.inputName]: this.inputTensor! },
      { [this.outputName]: this.outputTensor! },
    );

    const post = d.createCommandEncoder();
    this.ops!.encodeToRGBA(post, this.outNCHW!, this.outPixels!, this.size, this.size);
    post.copyBufferToBuffer(this.outPixels!, 0, this.readback!, 0, this.size * this.size * 4);
    d.queue.submit([post.finish()]);

    await this.readback!.mapAsync(GPUMapMode.READ);
    const out = new Uint8ClampedArray(this.readback!.getMappedRange().slice(0));
    this.readback!.unmap();
    return out;
  }

  private ensureImageBuffers(w: number, h: number) {
    if (this.imgW === w && this.imgH === h && this.imgPixels) return;
    [this.imgPixels, this.accum, this.weight, this.outPixelsImg, this.imgReadback]
      .forEach((b) => b?.destroy());
    const d = this.device;
    const px = w * h;
    this.imgPixels = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.accum = d.createBuffer({ size: 3 * px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.weight = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.outPixelsImg = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.imgReadback = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.imgW = w;
    this.imgH = h;
  }

  /**
   * Denoise a full-resolution image by tiling into 256² tiles (overlap 32, stride 224),
   * running each tile and blending with the min-of-sigmoid mask — the GPU equivalent of
   * the old tiler.ts. Everything stays on the device except the final pixel readback.
   */
  async denoiseImage(rgba: Uint8ClampedArray, w: number, h: number): Promise<Uint8ClampedArray> {
    if (rgba.length !== w * h * 4) throw new Error(`Expected ${w * h * 4} RGBA bytes, got ${rgba.length}`);
    if (!this.ops) this.setupGpuPath();
    this.ensureImageBuffers(w, h);
    const d = this.device;
    const TILE = this.size;
    const stride = TILE - this.overlap;
    const tilesX = Math.ceil(w / stride);
    const tilesY = Math.ceil(h / stride);

    // See ArrayBuffer-vs-ArrayBufferLike note above.
    d.queue.writeBuffer(this.imgPixels!, 0, rgba as BufferSource);
    const clr = d.createCommandEncoder();
    clr.clearBuffer(this.accum!);
    clr.clearBuffer(this.weight!);
    d.queue.submit([clr.finish()]);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const startX = tx * stride;
        const startY = ty * stride;
        const curW = Math.min(TILE, w - startX);
        const curH = Math.min(TILE, h - startY);

        const e1 = d.createCommandEncoder();
        this.ops!.encodeExtractTile(e1, this.imgPixels!, this.nchwInput!, w, h, startX, startY, TILE);
        d.queue.submit([e1.finish()]);

        await this.session.run(
          { [this.inputName]: this.inputTensor! },
          { [this.outputName]: this.outputTensor! },
        );

        const e2 = d.createCommandEncoder();
        this.ops!.encodeAccumulateTile(e2, this.outNCHW!, this.accum!, this.weight!, {
          imgW: w, imgH: h, startX, startY, curW, curH,
          tileX: tx, tileY: ty, tilesX, tilesY, tile: TILE, overlap: this.overlap,
        });
        d.queue.submit([e2.finish()]);
      }
    }

    const e3 = d.createCommandEncoder();
    this.ops!.encodeResolve(e3, this.accum!, this.weight!, this.outPixelsImg!, w, h);
    e3.copyBufferToBuffer(this.outPixelsImg!, 0, this.imgReadback!, 0, w * h * 4);
    d.queue.submit([e3.finish()]);

    await this.imgReadback!.mapAsync(GPUMapMode.READ);
    const out = new Uint8ClampedArray(this.imgReadback!.getMappedRange().slice(0));
    this.imgReadback!.unmap();
    return out;
  }

  tileGrid(w: number, h: number): { tilesX: number; tilesY: number; total: number } {
    const stride = this.size - this.overlap;
    const tilesX = Math.ceil(w / stride);
    const tilesY = Math.ceil(h / stride);
    return { tilesX, tilesY, total: tilesX * tilesY };
  }
}

/** HWC RGBA uint8 (canvas ImageData) -> normalized NCHW float32 (drops alpha). */
export function rgbaToNCHW(rgba: Uint8ClampedArray, w: number, h: number, channels = 3): Float32Array {
  const out = new Float32Array(channels * w * h);
  const plane = w * h;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      for (let c = 0; c < channels; c++) out[c * plane + p] = rgba[p * 4 + c] / 255;
    }
  }
  return out;
}

/** NCHW float32 (3 ch, [0,1]) -> RGBA uint8 for putImageData. */
export function nchwToRGBA(nchw: Float32Array, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  const plane = w * h;
  for (let p = 0; p < plane; p++) {
    out[p * 4 + 0] = nchw[0 * plane + p] * 255;
    out[p * 4 + 1] = nchw[1 * plane + p] * 255;
    out[p * 4 + 2] = nchw[2 * plane + p] * 255;
    out[p * 4 + 3] = 255;
  }
  return out;
}
