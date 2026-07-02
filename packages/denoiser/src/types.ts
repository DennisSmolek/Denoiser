export type Quality = 'fast' | 'balanced' | 'high';

export type InputName = 'color' | 'albedo' | 'normal';

export type ImgInput =
  | ImageData
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageBitmap
  | OffscreenCanvas;

export type OutputMode = 'imgData' | 'float32' | 'gpuTexture';

export type ListenerCallback = (result: unknown) => void;

export interface DenoiserProps {
  filterType: string; // 'rt'
  quality: Quality;
  hdr: boolean;
  srgb: boolean;
  height: number;
  width: number;
  cleanAux: boolean;
  dirtyAux: boolean;
  directionals: boolean;
  useColor: boolean;
  useAlbedo: boolean;
  useNormal: boolean;
}

export interface DenoiserOptions {
  /** Where ORT loads its wasm/jsep assets (default: jsDelivr CDN). */
  wasmPaths?: string;
  /** Use fp16 models when available (smaller/faster; needs the shader-f16 feature). */
  precision?: 'fp32' | 'fp16';
  /** Opt-in ORT WebGPU graph capture (unstable in onnxruntime-web 1.27 at high tile counts). */
  graphCapture?: boolean;
  /** Tiles per model run (default 8). Higher = fewer sync points, more GPU memory. */
  batch?: number;
}
