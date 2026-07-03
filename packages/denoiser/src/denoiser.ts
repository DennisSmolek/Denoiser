import { Models } from './weights';
import { DenoiseEngine } from './ort/engine';
import { determineModel } from './modelName';
import { imgToRGBA, getCorrectImageData, hasSizeMissmatch } from './utils';
import {
  DenoiserCreateOptions, DenoiseImageOptions, DenoiseTexturesOptions, DenoiserEvent,
  DenoiserInputError, Quality,
} from './types';
import type { DenoiseStats } from './ort/engine';

/**
 * Browser OIDN denoiser running fully on WebGPU via onnxruntime-web (v2 API).
 *
 * Execution is stateless per call — everything about a run is in the call's
 * options. The instance owns identity only: models, sessions, and the shared
 * GPUDevice (ORT creates it; read `denoiser.device` to share with three.js).
 *
 * ```ts
 * const denoiser = await Denoiser.create({ precision: 'fp16' });
 * const img = await denoiser.denoise(noisyImage);                    // ImageData
 * const tex = await denoiser.denoiseTextures({ color, hdr: true });  // GPUTexture
 * ```
 */
export class Denoiser {
  /** The shared GPUDevice ORT created — pass to three.js WebGPURenderer. */
  device!: GPUDevice;

  private models: Models;
  private engine!: DenoiseEngine;
  private activeModelName?: string;
  private aborted = false;
  private opts: Required<Pick<DenoiserCreateOptions, 'quality'>> & DenoiserCreateOptions;
  private listeners = new Map<DenoiserEvent, Set<(v: unknown) => void>>();

  /** Per-stage wall-clock timings from the most recent run. */
  get stats(): DenoiseStats | undefined { return this.engine?.lastStats; }
  get quality(): Quality { return this.opts.quality; }
  set quality(q: Quality) { this.opts.quality = q; }

  private constructor(opts: DenoiserCreateOptions) {
    this.opts = { quality: 'fast', ...opts };
    this.models = Models.getInstance();
    if (opts.precision) this.models.precision = opts.precision;
    if (opts.weightsUrl) this.models.url = opts.weightsUrl;
  }

  /** Async construction: loads the default model and creates the GPUDevice. */
  static async create(opts: DenoiserCreateOptions = {}): Promise<Denoiser> {
    const d = new Denoiser(opts);
    await d.ensureEngine({ hdr: false, albedo: false, normal: false });
    return d;
  }

  // ---- execution ----------------------------------------------------------

  /** Denoise an image-like input. Returns ImageData (undefined when aborted). */
  async denoise(color: ImgInputArg, options: DenoiseImageOptions = {}): Promise<ImageData | undefined> {
    const c = toRGBA(color);
    const albedo = options.albedo !== undefined ? toRGBA(options.albedo) : undefined;
    const normal = options.normal !== undefined ? toRGBA(options.normal) : undefined;
    if ((albedo && sizeDiffers(albedo, c)) || (normal && sizeDiffers(normal, c))) {
      throw new DenoiserInputError('aux inputs must match the color input size');
    }
    await this.ensureEngine({ hdr: false, albedo: !!albedo, normal: !!normal });
    this.aborted = false;
    const srgb = options.srgb ?? true; // photographs/screens are sRGB-encoded
    const out = await this.engine.denoise(c.data, c.width, c.height, {
      albedo: albedo?.data,
      normal: normal?.data,
      // RGBA8 bytes are display-encoded; upstream treats that as identity
      // (srgb -> Linear). srgb:false means "my bytes are linear" -> encode.
      srgb: !srgb,
      hdr: false,
      flipY: options.flipY,
      onProgress: this.progressCb(options.onProgress),
    });
    if (this.aborted) return this.emitExecuted(undefined);
    const img = new ImageData(out, c.width, c.height);
    this.emitExecuted(img);
    return img;
  }

  /** Denoise, returning normalized RGBA floats instead of ImageData. */
  async denoiseToFloat(color: ImgInputArg, options: DenoiseImageOptions = {}): Promise<Float32Array | undefined> {
    const img = await this.denoise(color, options);
    if (!img) return undefined;
    const f = new Float32Array(img.data.length);
    for (let i = 0; i < f.length; i++) f[i] = img.data[i] / 255;
    return f;
  }

  /**
   * Zero-copy denoise: float textures in, GPUTexture out. With `output` the
   * result lands in YOUR texture (rgba8unorm / rgba16float, STORAGE_BINDING);
   * otherwise an engine-owned rgba8unorm texture is returned (valid until the
   * next call / size change / teardown). Returns undefined when aborted.
   */
  async denoiseTextures(options: DenoiseTexturesOptions): Promise<GPUTexture | undefined> {
    await this.ensureEngine({
      hdr: !!options.hdr, albedo: !!options.albedo, normal: !!options.normal,
    });
    this.aborted = false;
    const transfer = options.transfer ?? 'linear';
    const out = await this.engine.denoiseTextures(
      { color: options.color, albedo: options.albedo, normal: options.normal },
      {
        hdr: options.hdr,
        inputScale: options.inputScale,
        srgb: false, // float texture inputs are linear
        tonemap: transfer === 'aces-srgb',
        // resolve's srgb flag = encode output; the extract side ignores it for hdr
        ...(transfer === 'srgb' ? { srgb: true } : {}),
        inputFlipY: options.inputFlipY,
        auxInputFlipY: options.auxInputFlipY,
        flipY: options.outputFlipY,
        toTexture: true,
        outputTexture: options.output,
        onProgress: this.progressCb(options.onProgress),
      });
    const tex = this.aborted ? undefined : (out as GPUTexture);
    this.emitExecuted(tex);
    return tex;
  }

  /** Drop the in-flight run's result (its GPU work still completes). */
  abort() { this.aborted = true; }

  // ---- lifecycle ----------------------------------------------------------

  /**
   * Release image buffers and extra sessions but KEEP the shared GPUDevice
   * alive (one model session is retained). Safe to call between workloads.
   */
  dispose() { this.engine?.trim(); }

  /**
   * Full teardown. Releasing the last ORT session DESTROYS the shared
   * GPUDevice — three.js renderers and canvases on it die too. Only call
   * this when the whole WebGPU stack is going away.
   */
  destroyDevice() { this.engine?.destroy(); }

  // ---- events -------------------------------------------------------------

  on(event: DenoiserEvent, cb: (value: unknown) => void): () => void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(cb);
    return () => this.listeners.get(event)?.delete(cb);
  }

  // ---- internals ----------------------------------------------------------

  private progressCb(local?: (p: number) => void) {
    const set = this.listeners.get('progress');
    if (!local && !set?.size) return undefined;
    return (p: number) => {
      local?.(p);
      set?.forEach((cb) => cb(p));
    };
  }

  private emitExecuted<T>(value: T): T {
    this.listeners.get('executed')?.forEach((cb) => cb(value));
    return value;
  }

  /** (Re)build the engine when the required model changes. Overlaps creation
   *  with disposal so the ORT session count never hits zero (device survives). */
  private async ensureEngine(sel: { hdr: boolean; albedo: boolean; normal: boolean }) {
    const { name, channels } = determineModel({
      filterType: 'rt', quality: this.opts.quality, hdr: sel.hdr,
      useColor: true, useAlbedo: sel.albedo, useNormal: sel.normal,
      cleanAux: sel.albedo && sel.normal, dirtyAux: false,
    });
    if (this.engine && this.activeModelName === name) return;
    const create = async () =>
      DenoiseEngine.create(await this.models.get(name), {
        channels,
        wasmPaths: this.opts.wasmPaths,
        graphCapture: this.opts.graphCapture,
        batch: this.opts.batch,
        maxRunPixels: this.opts.maxRunPixels,
        precision: this.models.precision,
      });
    const old = this.engine;
    try {
      this.engine = await create();
    } catch (err) {
      if (this.models.precision !== 'fp16') throw err;
      console.warn('Denoiser: fp16 unavailable, falling back to fp32', err);
      this.models.precision = 'fp32';
      this.engine = await create();
    }
    old?.destroy();
    this.activeModelName = name;
    this.device = this.engine.device;
  }
}

type ImgInputArg = Parameters<typeof toRGBA>[0];

function toRGBA(input: import('./types').ImgInput): { data: Uint8ClampedArray; width: number; height: number } {
  if (input && typeof input === 'object' && 'data' in input && input.data instanceof Uint8ClampedArray
    && !(input instanceof ImageData)) {
    const raw = input as { data: Uint8ClampedArray; width: number; height: number };
    if (raw.data.length !== raw.width * raw.height * 4) {
      throw new DenoiserInputError('raw input must be RGBA8 (width*height*4 bytes)');
    }
    return raw;
  }
  let source = input as Exclude<import('./types').ImgInput, { data: Uint8ClampedArray; width: number; height: number }> | ImageData;
  if (source instanceof HTMLImageElement && hasSizeMissmatch(source)) {
    source = getCorrectImageData(source);
  }
  return imgToRGBA(source as Parameters<typeof imgToRGBA>[0]);
}

function sizeDiffers(a: { width: number; height: number }, b: { width: number; height: number }) {
  return a.width !== b.width || a.height !== b.height;
}
