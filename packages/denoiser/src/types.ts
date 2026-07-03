export type Quality = 'fast' | 'balanced' | 'high';

export type ImgInput =
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageBitmap
  | OffscreenCanvas
  | { data: Uint8ClampedArray; width: number; height: number };

/** Output encoding for texture results. 'linear' = raw model output (HDR-safe),
 *  'srgb' = sRGB-encode, 'aces-srgb' = ACES tonemap + sRGB (display-ready). */
export type OutputTransfer = 'linear' | 'srgb' | 'aces-srgb';

export interface DenoiserCreateOptions {
  /** fp16 models + tensors when the device supports shader-f16 (auto-falls back). */
  precision?: 'fp32' | 'fp16';
  /** Model family size: fast = *_small, balanced = base, high = *_large where available. */
  quality?: Quality;
  /** Where the .onnx models are served (default: jsDelivr CDN). */
  weightsUrl?: string;
  /** Where ORT loads its wasm assets (default: jsDelivr CDN). */
  wasmPaths?: string;
  /** Per-run pixel budget: images above it tile instead of whole-frame. Default 2048*1152. */
  maxRunPixels?: number;
  /** Max tiles per model run in tiled mode (default 8). */
  batch?: number;
  /** Opt-in ORT WebGPU graph capture (unstable in onnxruntime-web 1.27 past ~150 replays). */
  graphCapture?: boolean;
}

export interface DenoiseImageOptions {
  albedo?: ImgInput; // [0,1]
  normal?: ImgInput; // RGBA8 bytes encoding [-1,1] as [0,1]
  /** Input pixels are sRGB-encoded photographs/screens (decode in, re-encode out). Default true. */
  srgb?: boolean;
  /** Flip the result vertically. */
  flipY?: boolean;
  onProgress?: (p: number) => void;
}

export interface DenoiseTexturesOptions {
  color: GPUTexture; // float, linear (HDR or LDR)
  albedo?: GPUTexture; // [0,1] floats
  normal?: GPUTexture; // [-1,1] floats (encoded for the network internally)
  /** Linear-HDR input: applies OIDN's PU transfer + autoexposure around the model. */
  hdr?: boolean;
  /** Manual HDR input scale (overrides autoexposure). */
  inputScale?: number;
  /** Color texture rows are bottom-up (WebGPU render targets) — flip reads. */
  inputFlipY?: boolean;
  /** Aux flip when their vertical convention differs from color. Default: inputFlipY. */
  auxInputFlipY?: boolean;
  /** Caller-owned output texture (rgba8unorm or rgba16float, STORAGE_BINDING). */
  output?: GPUTexture;
  /** Output encoding (default 'linear'; rgba8unorm outputs clamp regardless). */
  transfer?: OutputTransfer;
  /** Flip the result vertically. */
  outputFlipY?: boolean;
  onProgress?: (p: number) => void;
}

export type DenoiserEvent = 'progress' | 'executed';

/** WebGPU / shader-f16 / device capability problems. */
export class DenoiserUnsupportedError extends Error {
  constructor(message: string) { super(message); this.name = 'DenoiserUnsupportedError'; }
}
/** Bad or inconsistent inputs (sizes, formats, missing aux). */
export class DenoiserInputError extends Error {
  constructor(message: string) { super(message); this.name = 'DenoiserInputError'; }
}
