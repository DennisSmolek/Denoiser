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
   * Opt-in WebGPU graph capture. Measured ~0 gain here (the workload is
   * GPU-bound, not dispatch-bound) and onnxruntime-web 1.27.0 captured
   * sessions crash after ~150-250 CUMULATIVE replays (createBindGroup:
   * "Required member is undefined" in ORT's capture buffer manager; GPU
   * syncs don't prevent it — standalone repro in the
   * ort-webgpu-graphcapture-repro repo). Off by default until upstream fixes.
   */
  graphCapture?: boolean;
  /**
   * Aux split-graph workaround for the onnxruntime-web WebGPU Conv bug (the
   * first conv reducing the raw >3ch input miscomputes — see
   * tools/ort-webgpu-aux-repro). When set, `modelBytes` is a re-exported TAIL
   * model that starts at `enc_conv1` and takes TWO inputs: the enc_conv0 feature
   * map AND the raw `input` (the dec_conv1a skip still needs it). We compute
   * enc_conv0 (Conv 3x3 pad1 CIN->encOutChannels + relu6) ourselves in WGSL and
   * feed both. Verified to restore native quality (1.2e-6 vs reference).
   */
  split?: {
    encWeights: Float32Array; // OIHW [encOutChannels, channels, 3, 3]
    encBias: Float32Array; // [encOutChannels]
    encOutChannels: number; // enc_conv0 output channels (32 for the OIDN aux nets)
    featInputName?: string; // tail input for the feature map (default 'enc_conv0_relu6_2')
    rawInputName?: string; // tail input for the raw image (default 'input')
  };
}

export interface DenoiseOptions {
  albedo?: Uint8ClampedArray; // required when channels >= 6
  normal?: Uint8ClampedArray; // required when channels >= 9
  srgb?: boolean; // input is sRGB -> convert to linear before the model, back after
  hdr?: boolean; // linear-HDR input: OIDN PU transfer + autoexposure applied around the model
  /** Manual HDR input scale (overrides autoexposure; OIDN semantics). */
  inputScale?: number;
  flipY?: boolean; // flip the output vertically (in the resolve kernel, free)
  /** ACES tonemap + sRGB-encode the output (for HDR results going straight to a canvas). */
  tonemap?: boolean;
  onProgress?: (p: number) => void;
}

/** Zero-copy input: float textures living on the shared device (e.g. render targets). */
export interface TextureInputs {
  color: GPUTexture;
  albedo?: GPUTexture; // [0,1] floats
  normal?: GPUTexture; // [-1,1] floats (G-buffer convention); encoded to [0,1] for the network
}

export interface TextureDenoiseOptions extends DenoiseOptions {
  /** Source textures are bottom-up (WebGPU render targets) — flip reads. */
  inputFlipY?: boolean;
  /** Aux textures' vertical convention when it differs from color (e.g. raster
   *  G-buffer vs compute-written tracer output). Defaults to inputFlipY. */
  auxInputFlipY?: boolean;
  /** Resolve into an engine-owned rgba8unorm GPUTexture and return it (no CPU readback). */
  toTexture?: boolean;
  /**
   * Resolve into a CALLER-owned texture instead (e.g. a three.js StorageTexture's
   * GPUTexture) — the integration path: pathtracer -> denoiser -> render target.
   * Must match the image size, have STORAGE_BINDING usage, and be rgba8unorm
   * (clamped/display-ready) or rgba16float (unclamped, keeps HDR range so the
   * renderer's own tonemapping stays in charge). Implies toTexture.
   */
  outputTexture?: GPUTexture;
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
  encFeat?: GPUBuffer; // split mode: enc_conv0 output [B, encOutChannels, H, W]
  encTensor?: ort.Tensor; // split mode: tail feature-map input tensor over encFeat
}

/** Split-graph workaround state (see EngineOptions.split). */
interface SplitState {
  encWeights: Float32Array;
  encBias: Float32Array;
  encOutChannels: number;
  featInputName: string;
  rawInputName: string;
  weightBuf?: GPUBuffer; // uploaded once (device-lifetime)
  biasBuf?: GPUBuffer;
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
  private split?: SplitState;

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
  private outTexture?: GPUTexture;

  private constructor(opts: EngineOptions) {
    this.channels = opts.channels;
    this.tile = opts.tile ?? 256;
    this.overlap = opts.overlap ?? 32;
    this.batch = Math.max(1, opts.batch ?? 8);
    this.precision = opts.precision ?? 'fp32';
    this.maxRunPixels = opts.maxRunPixels ?? 2048 * 1152;
    if (opts.split) {
      const s = opts.split;
      const expectW = s.encOutChannels * this.channels * 9;
      if (s.encWeights.length !== expectW) {
        throw new Error(`Denoiser: split encWeights must be ${expectW} floats ([${s.encOutChannels},${this.channels},3,3]), got ${s.encWeights.length}`);
      }
      if (s.encBias.length !== s.encOutChannels) {
        throw new Error(`Denoiser: split encBias must be ${s.encOutChannels} floats, got ${s.encBias.length}`);
      }
      this.split = {
        encWeights: s.encWeights,
        encBias: s.encBias,
        encOutChannels: s.encOutChannels,
        featInputName: s.featInputName ?? 'enc_conv0_relu6_2',
        rawInputName: s.rawInputName ?? 'input',
      };
    }
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
      e.destroy();
      throw new Error('Denoiser: fp16 needs the shader-f16 WebGPU feature (unavailable on this device)');
    }
    e.ops = new GpuImageOps(e.device, e.batch, e.precision === 'fp16');

    // Split mode: upload enc_conv0 weights/bias once (device-lifetime). f32 for
    // accuracy even when the model IO is fp16 (accumulation is f32 in the kernel).
    if (e.split) {
      const s = e.split;
      s.weightBuf = e.device.createBuffer({
        size: s.encWeights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      s.biasBuf = e.device.createBuffer({
        size: s.encBias.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      e.device.queue.writeBuffer(s.weightBuf, 0, s.encWeights);
      e.device.queue.writeBuffer(s.biasBuf, 0, s.encBias);
    }
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

    this.outputName = session.outputNames[0];
    if (this.split) {
      // Tail model has two inputs: the enc_conv0 feature map and the raw image.
      for (const n of [this.split.featInputName, this.split.rawInputName]) {
        if (!session.inputNames.includes(n)) {
          throw new Error(`Denoiser: split tail model missing input '${n}' (has: ${session.inputNames.join(', ')})`);
        }
      }
      this.inputName = this.split.rawInputName; // the raw image feed
    } else {
      this.inputName = session.inputNames[0];
    }
    const device = (ort.env.webgpu as unknown as { device: GPUDevice }).device;
    const els = plan.batch * plan.tileW * plan.tileH;
    const f16 = this.precision === 'fp16';
    const dtype = f16 ? 'float16' : 'float32';
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
        dataType: dtype,
        dims: [plan.batch, this.channels, plan.tileH, plan.tileW],
      }),
      outputTensor: ort.Tensor.fromGpuBuffer(outNCHW, {
        dataType: dtype,
        dims: [plan.batch, 3, plan.tileH, plan.tileW],
      }),
    };
    if (this.split) {
      const cout = this.split.encOutChannels;
      geo.encFeat = device.createBuffer({
        size: els * cout * this.bpe,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      geo.encTensor = ort.Tensor.fromGpuBuffer(geo.encFeat, {
        dataType: dtype,
        dims: [plan.batch, cout, plan.tileH, plan.tileW],
      });
    }
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
    g.encFeat?.destroy();
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

  private ensureImageBuffers(w: number, h: number, cpuInput: boolean) {
    const haveInputs = !cpuInput || !!this.color;
    if (this.imgW === w && this.imgH === h && this.accum && haveInputs) return;
    [this.color, this.albedo, this.normal, this.accum, this.weight, this.outPixels, this.readback]
      .forEach((b) => b?.destroy());
    this.outTexture?.destroy();
    this.outTexture = undefined;
    const d = this.device;
    const px = w * h;
    const stor = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    if (cpuInput) {
      this.color = d.createBuffer({ size: px * 4, usage: stor });
      this.albedo = this.channels >= 6 ? d.createBuffer({ size: px * 4, usage: stor }) : undefined;
      this.normal = this.channels >= 9 ? d.createBuffer({ size: px * 4, usage: stor }) : undefined;
    } else {
      this.color = this.albedo = this.normal = undefined;
    }
    this.accum = d.createBuffer({ size: 3 * px * 4, usage: stor });
    this.weight = d.createBuffer({ size: px * 4, usage: stor });
    this.outPixels = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.readback = d.createBuffer({ size: px * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.imgW = w;
    this.imgH = h;
  }

  private ensureOutTexture(w: number, h: number): GPUTexture {
    if (!this.outTexture || this.outTexture.width !== w || this.outTexture.height !== h) {
      this.outTexture?.destroy();
      this.outTexture = this.device.createTexture({
        size: { width: w, height: h },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
      });
    }
    return this.outTexture;
  }

  /** Denoise a full-resolution image (whole-frame or tiled+blended). Returns RGBA8 pixels (alpha = 255). */
  async denoise(color: Uint8ClampedArray, w: number, h: number, opts: DenoiseOptions = {}): Promise<Uint8ClampedArray> {
    if (color.length !== w * h * 4) throw new Error(`Denoiser: expected ${w * h * 4} color bytes, got ${color.length}`);
    if (this.channels >= 6 && !opts.albedo) throw new Error('Denoiser: model requires an albedo input');
    if (this.channels >= 9 && !opts.normal) throw new Error('Denoiser: model requires a normal input');
    return this.process({ cpu: { color, albedo: opts.albedo, normal: opts.normal } }, w, h, opts) as Promise<Uint8ClampedArray>;
  }

  /**
   * Zero-copy denoise: read float input textures on the shared device directly
   * (no CPU round-trip, no 8-bit quantization — feed HDR models real HDR).
   * With toTexture, also skips the readback and returns an rgba8unorm texture
   * (owned by the engine, valid until the next call / size change / dispose).
   */
  async denoiseTextures(inputs: TextureInputs, opts: TextureDenoiseOptions = {}): Promise<Uint8ClampedArray | GPUTexture> {
    if (this.channels >= 6 && !inputs.albedo) throw new Error('Denoiser: model requires an albedo input');
    if (this.channels >= 9 && !inputs.normal) throw new Error('Denoiser: model requires a normal input');
    return this.process({ tex: inputs }, inputs.color.width, inputs.color.height, opts);
  }

  private async process(
    src: { cpu?: { color: Uint8ClampedArray; albedo?: Uint8ClampedArray; normal?: Uint8ClampedArray }; tex?: TextureInputs },
    w: number, h: number, opts: TextureDenoiseOptions,
  ): Promise<Uint8ClampedArray | GPUTexture> {
    const geo = await this.ensureGeo(this.planFor(w, h));
    const { tileW, tileH, batch: B, overlap } = geo.plan;

    this.ensureImageBuffers(w, h, !!src.cpu);
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
    if (src.cpu) {
      d.queue.writeBuffer(this.color!, 0, src.cpu.color);
      if (src.cpu.albedo) d.queue.writeBuffer(this.albedo!, 0, src.cpu.albedo);
      if (src.cpu.normal) d.queue.writeBuffer(this.normal!, 0, src.cpu.normal);
    }

    const clr = d.createCommandEncoder();
    clr.clearBuffer(this.accum!);
    clr.clearBuffer(this.weight!);
    // HDR input scale (OIDN semantics): manual value, or autoexposure computed
    // on the GPU from the color texture. 8-bit inputs default to scale 1.
    if (opts.hdr && opts.inputScale === undefined && src.tex) {
      this.ops.encodeAutoexposure(clr, src.tex.color.createView(), w, h);
    } else {
      this.ops.setExposure(opts.inputScale ?? 1);
    }
    d.queue.submit([clr.finish()]);
    const tUpload = performance.now();

    const albedoBuf = this.albedo ?? this.color;
    const normalBuf = this.normal ?? this.color;
    const colorView = src.tex?.color.createView();
    const albedoView = src.tex ? (src.tex.albedo ?? src.tex.color).createView() : undefined;
    const normalView = src.tex ? (src.tex.normal ?? src.tex.color).createView() : undefined;
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
      if (src.tex) {
        this.ops.encodeExtractTilesTex(
          e1, colorView!, albedoView!, normalView!, geo.nchwInput,
          w, h, tileW, tileH, this.channels, !!opts.srgb, !!opts.inputFlipY, !!opts.hdr,
          !!(opts.auxInputFlipY ?? opts.inputFlipY), offsets, chunk.length);
      } else {
        this.ops.encodeExtractTiles(
          e1, this.color!, albedoBuf!, normalBuf!, geo.nchwInput,
          w, h, tileW, tileH, this.channels, !!opts.srgb, !!opts.hdr, offsets, chunk.length);
      }
      // Split mode: compute enc_conv0 (nchwInput -> encFeat) ourselves, then run
      // the tail with BOTH the feature map and the raw input (dec_conv1a skip).
      if (this.split) {
        this.ops.encodeEncConv0(
          e1, geo.nchwInput, this.split.weightBuf!, this.split.biasBuf!, geo.encFeat!,
          tileW, tileH, this.channels, this.split.encOutChannels, B);
      }
      d.queue.submit([e1.finish()]);
      const t1 = performance.now();
      encodeMs += t1 - t0;

      // Unused slots of a short final batch still run through the model with
      // stale (valid float) contents; their outputs are simply never blended.
      const feeds = this.split
        ? {
            [this.split.featInputName]: geo.encTensor!,
            [this.split.rawInputName]: geo.inputTensor,
          }
        : { [this.inputName]: geo.inputTensor };
      await geo.session.run(feeds, { [this.outputName]: geo.outputTensor });
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
    let out: Uint8ClampedArray | GPUTexture;
    if (opts.outputTexture || opts.toTexture) {
      let tex = opts.outputTexture;
      if (tex) {
        if (tex.width !== w || tex.height !== h) {
          throw new Error(`Denoiser: outputTexture is ${tex.width}x${tex.height}, image is ${w}x${h}`);
        }
        if (!(tex.usage & GPUTextureUsage.STORAGE_BINDING)) {
          throw new Error('Denoiser: outputTexture needs STORAGE_BINDING usage');
        }
        if (tex.format !== 'rgba8unorm' && tex.format !== 'rgba16float') {
          throw new Error(`Denoiser: outputTexture must be rgba8unorm or rgba16float (got ${tex.format})`);
        }
      } else {
        tex = this.ensureOutTexture(w, h);
      }
      const e3 = d.createCommandEncoder();
      this.ops.encodeResolveToTexture(
        e3, this.accum!, this.weight!, tex.createView(), tex.format,
        w, h, !!opts.srgb, !!opts.hdr, !!opts.flipY, !!opts.tonemap);
      d.queue.submit([e3.finish()]);
      await d.queue.onSubmittedWorkDone();
      out = tex;
    } else {
      const e3 = d.createCommandEncoder();
      this.ops.encodeResolve(
        e3, this.accum!, this.weight!, this.outPixels!,
        w, h, !!opts.srgb, !!opts.hdr, !!opts.flipY, !!opts.tonemap);
      e3.copyBufferToBuffer(this.outPixels!, 0, this.readback!, 0, w * h * 4);
      d.queue.submit([e3.finish()]);

      await this.readback!.mapAsync(GPUMapMode.READ);
      out = new Uint8ClampedArray(this.readback!.getMappedRange().slice(0));
      this.readback!.unmap();
    }
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

  /** Free per-image buffers and all non-default geometry sessions. The default
   *  session (and therefore the shared GPUDevice) stays alive. */
  trim() {
    [this.color, this.albedo, this.normal, this.accum, this.weight, this.outPixels, this.readback]
      .forEach((b) => b?.destroy());
    this.color = this.albedo = this.normal = this.accum = this.weight =
      this.outPixels = this.readback = undefined;
    this.outTexture?.destroy();
    this.outTexture = undefined;
    this.imgW = this.imgH = 0;
    let first = true;
    for (const [key, g] of this.geos) {
      if (first) { first = false; continue; }
      this.releaseGeo(g);
      this.geos.delete(key);
    }
  }

  /** Full teardown. Releasing the last ORT session DESTROYS the shared
   *  GPUDevice — anything else using it (three.js, canvases) dies with it. */
  destroy() {
    this.trim();
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
