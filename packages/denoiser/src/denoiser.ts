import { Models } from './weights';
import { DenoiseEngine } from './ort/engine';
import { determineModel } from './modelName';
import { imgToRGBA, flipRGBAY, getCorrectImageData, hasSizeMissmatch, formatTime } from './utils';
import type {
  DenoiserProps, DenoiserOptions, ImgInput, InputName, ListenerCallback, OutputMode, Quality,
} from './types';

interface InputImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/**
 * Browser OIDN denoiser running fully on WebGPU via onnxruntime-web.
 *
 * Replaces the old TensorFlow.js + WebGL implementation. The U-Net runs on ORT's
 * WebGPU EP; normalization, layout, sRGB, tiling and overlap-blend are WGSL compute.
 * ORT owns the GPUDevice — read `denoiser.device` to share it with three.js's
 * WebGPURenderer (onnxruntime issue #26107).
 */
export class Denoiser {
  timesGenerated = 0;

  props: DenoiserProps = {
    filterType: 'rt',
    quality: 'fast',
    hdr: false,
    srgb: false,
    height: 0,
    width: 0,
    cleanAux: false,
    dirtyAux: false,
    directionals: false,
    useColor: true,
    useAlbedo: false,
    useNormal: false,
  };

  flipOutputY = false;
  outputMode: OutputMode = 'imgData';
  debugging = false;

  canvas?: HTMLCanvasElement;
  outputToCanvas = false;

  //* internal
  private models: Models;
  private engine?: DenoiseEngine;
  private activeModelName?: string;
  private isDirty = true;
  private aborted = false;
  private backendLoaded = false;

  private inputs: Map<InputName, InputImage> = new Map();

  private listeners: Map<ListenerCallback, OutputMode> = new Map();
  private backendListeners: Set<ListenerCallback> = new Set();
  private progressListeners: Set<(p: number) => void> = new Set();

  private wasmPaths?: string;
  private graphCapture = false;

  stats: Record<string, number | string> = {};
  private timers: Record<string, number> = {};

  constructor(opts: DenoiserOptions = {}) {
    this.models = Models.getInstance();
    this.wasmPaths = opts.wasmPaths;
    this.graphCapture = opts.graphCapture ?? false;
    if (opts.precision) this.models.precision = opts.precision;
    if (this.debugging) console.log('%c Denoiser initialized (WebGPU/ORT)', 'background: #d66b00; color: white;');
  }

  //* Getters / setters ------------------------------
  /** The GPUDevice ORT created — pass to three.js WebGPURenderer for zero-copy interop. */
  get device(): GPUDevice | undefined { return this.engine?.device; }

  /** Per-stage wall-clock timings from the most recent execution (for benchmarking). */
  get lastStats() { return this.engine?.lastStats; }

  /** Tile grid the engine would use for the current input size. */
  get tileInfo() {
    if (!this.engine) return undefined;
    const { tilesX, tilesY } = this.engine.tileGrid(this.props.width, this.props.height);
    return { tilesX, tilesY, tile: this.engine.tile, overlap: this.engine.overlap };
  }

  get backendReady() { return this.backendLoaded; }

  set weightsUrl(url: string) { this.models.url = url; }
  get weightsUrl() { return this.models.url; }
  set weightsPath(path: string) { this.models.path = path; }
  get weightsPath() { return this.models.path ?? ''; }

  get height() { return this.props.height; }
  set height(v: number) { this.setProp('height', v); }
  get width() { return this.props.width; }
  set width(v: number) { this.setProp('width', v); }
  get quality() { return this.props.quality; }
  set quality(v: Quality) { this.setProp('quality', v); }
  get hdr() { return this.props.hdr; }
  set hdr(v: boolean) { this.setProp('hdr', v); }
  get srgb() { return this.props.srgb; }
  set srgb(v: boolean) { this.props.srgb = v; } // pre/post only, no rebuild
  get dirtyAux() { return this.props.dirtyAux; }
  set dirtyAux(v: boolean) {
    if (v && this.props.cleanAux) this.props.cleanAux = false;
    this.setProp('dirtyAux', v);
  }

  private setProp<K extends keyof DenoiserProps>(key: K, value: DenoiserProps[K]) {
    if (this.props[key] === value) return;
    this.props[key] = value;
    this.isDirty = true;
  }

  //* Build ------------------------------------------
  async build() {
    const { name, channels } = determineModel(this.props);
    if (this.debugging) console.log(`Denoiser: building model ${name} (${channels}ch)`);

    if (this.engine && this.activeModelName === name) {
      this.isDirty = false;
      return;
    }
    this.engine?.dispose();
    this.startTimer('build');
    const bytes = await this.models.get(name);
    this.engine = await DenoiseEngine.create(bytes, {
      channels, wasmPaths: this.wasmPaths, graphCapture: this.graphCapture,
    });
    this.activeModelName = name;
    this.stopTimer('build');

    this.timesGenerated++;
    this.isDirty = false;
    this.backendLoaded = true;
    this.backendListeners.forEach((cb) => cb(this.device));
  }

  //* Execute ----------------------------------------
  async execute(colorInput?: ImgInput, albedoInput?: ImgInput, normalInput?: ImgInput) {
    if (colorInput) this.setInputImage('color', colorInput);
    if (albedoInput) this.setInputImage('albedo', albedoInput);
    if (normalInput) this.setInputImage('normal', normalInput);
    return this.executeModel();
  }

  private async executeModel() {
    this.aborted = false;
    this.startTimer('execution');

    const color = this.inputs.get('color');
    if (!color) throw new Error('Denoiser: a color input must be set before execution.');

    // aux presence determines the model; if it changed, rebuild
    this.props.useAlbedo = this.inputs.has('albedo');
    this.props.useNormal = this.inputs.has('normal');
    if (this.props.useAlbedo && this.props.useNormal && !this.props.dirtyAux) this.props.cleanAux = true;

    const { name } = determineModel(this.props);
    if (this.isDirty || !this.engine || this.activeModelName !== name) await this.build();
    if (this.aborted) return this.finishAbort();

    const albedo = this.inputs.get('albedo');
    const normal = this.inputs.get('normal');

    this.startTimer('inference');
    const out = await this.engine!.denoise(color.data, color.width, color.height, {
      albedo: albedo?.data,
      normal: normal?.data,
      srgb: this.props.srgb,
      hdr: this.props.hdr,
      onProgress: (p) => this.progressListeners.forEach((l) => l(p)),
    });
    this.stopTimer('inference');

    if (this.aborted) return this.finishAbort();
    const finalRGBA = this.flipOutputY ? flipRGBAY(out, color.width, color.height) : out;
    return this.handleReturn(finalRGBA, color.width, color.height);
  }

  private async handleReturn(rgba: Uint8ClampedArray, width: number, height: number) {
    if (this.outputToCanvas && this.canvas) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.canvas.getContext('2d')?.putImageData(new ImageData(rgba, width, height), 0, 0);
    }
    // listeners with their own response types
    this.listeners.forEach((mode, cb) => cb(this.format(rgba, width, height, mode)));
    // direct return only when no listeners
    let toReturn: unknown;
    if (this.listeners.size === 0) toReturn = this.format(rgba, width, height, this.outputMode);

    this.stopTimer('execution');
    this.logStats();
    return toReturn;
  }

  private format(rgba: Uint8ClampedArray, width: number, height: number, mode: OutputMode): unknown {
    if (mode === 'float32') {
      const f = new Float32Array(rgba.length);
      for (let i = 0; i < rgba.length; i++) f[i] = rgba[i] / 255;
      return f;
    }
    return new ImageData(rgba, width, height);
  }

  private finishAbort() {
    this.stopTimer('execution');
    return undefined;
  }

  //* Inputs -----------------------------------------
  setInputImage(name: InputName, imgData: ImgInput, flipY = false) {
    if (!imgData) throw new Error('Denoiser: no image data provided');
    let source = imgData;
    if (imgData instanceof HTMLImageElement && hasSizeMissmatch(imgData)) {
      source = getCorrectImageData(imgData);
    }
    const { data, width, height } = imgToRGBA(source);
    const finalData = flipY ? flipRGBAY(data, width, height) : data;

    if (name === 'color') {
      if (width !== this.props.width || height !== this.props.height) {
        this.props.width = width;
        this.props.height = height;
      }
      this.props.useColor = true;
    } else if (name === 'albedo') {
      this.props.useAlbedo = true;
      this.isDirty = true;
    } else if (name === 'normal') {
      this.props.useNormal = true;
      this.isDirty = true;
    }
    this.inputs.set(name, { data: finalData, width, height });
  }

  /** Set raw RGBA8 (or already-shaped) pixel data directly. */
  setInputData(name: InputName, data: Uint8ClampedArray, width: number, height: number) {
    if (data.length !== width * height * 4) throw new Error('Denoiser: data must be RGBA8 (width*height*4)');
    if (name === 'color') { this.props.width = width; this.props.height = height; this.props.useColor = true; }
    else if (name === 'albedo') { this.props.useAlbedo = true; this.isDirty = true; }
    else if (name === 'normal') { this.props.useNormal = true; this.isDirty = true; }
    this.inputs.set(name, { data, width, height });
  }

  resetInputs() {
    this.inputs.clear();
    this.props.useColor = false;
    this.props.useAlbedo = false;
    this.props.useNormal = false;
    this.isDirty = true;
  }

  setCanvas(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.outputToCanvas = true;
  }

  abort() {
    this.aborted = true;
  }

  dispose() {
    this.engine?.dispose();
    this.engine = undefined;
    this.resetInputs();
  }

  //* Listeners --------------------------------------
  onExecute(listener: ListenerCallback, responseType: OutputMode = this.outputMode) {
    this.listeners.set(listener, responseType);
    return () => this.listeners.delete(listener);
  }

  onProgress(listener: (progress: number) => void) {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  onBackendReady(listener: ListenerCallback) {
    if (this.backendReady) listener(this.device);
    else this.backendListeners.add(listener);
    return () => this.backendListeners.delete(listener);
  }

  //* Debug ------------------------------------------
  startTimer(name: string) { if (this.debugging) this.timers[`${name}In`] = performance.now(); }
  stopTimer(name: string) {
    if (!this.debugging) return;
    this.timers[`${name}Out`] = performance.now();
    this.stats[name] = this.timers[`${name}Out`] - this.timers[`${name}In`];
  }
  logStats() {
    if (!this.debugging) return;
    const formatted = Object.entries(this.stats).reduce((acc, [k, v]) => {
      acc[k] = typeof v === 'string' ? v : formatTime(v);
      return acc;
    }, {} as Record<string, string>);
    console.table(formatted);
    this.stats = {};
  }
}
