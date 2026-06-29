import type { ImgInput } from './types';

//* Image helpers ----------------------------------

/** Decode any supported image input to RGBA8 bytes + dimensions via a 2D canvas. */
export function imgToRGBA(input: ImgInput): { data: Uint8ClampedArray; width: number; height: number } {
  if (input instanceof ImageData) {
    return { data: input.data, width: input.width, height: input.height };
  }
  let width = 0;
  let height = 0;
  if (input instanceof HTMLImageElement) {
    width = input.naturalWidth || input.width;
    height = input.naturalHeight || input.height;
  } else if (typeof HTMLVideoElement !== 'undefined' && input instanceof HTMLVideoElement) {
    width = input.videoWidth;
    height = input.videoHeight;
  } else {
    width = (input as { width: number }).width;
    height = (input as { height: number }).height;
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Denoiser: could not get 2D canvas context');
  ctx.drawImage(input as CanvasImageSource, 0, 0, width, height);
  const id = ctx.getImageData(0, 0, width, height);
  return { data: id.data, width, height };
}

/** Flip RGBA pixel rows vertically (OIDN/GL Y convention helper). */
export function flipRGBAY(data: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(data.length);
  const row = width * 4;
  for (let y = 0; y < height; y++) {
    const src = y * row;
    const dst = (height - 1 - y) * row;
    out.set(data.subarray(src, src + row), dst);
  }
  return out;
}

/** A css-scaled <img> reports display size; redraw to get the true pixels. */
export function getCorrectImageData(img: HTMLImageElement): ImageData {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not get canvas context');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

export function hasSizeMissmatch(img: HTMLImageElement): boolean {
  if (!img.naturalHeight || !img.naturalWidth) return true;
  return img.height !== img.naturalHeight || img.width !== img.naturalWidth;
}

//* General ----------------------------------------
export function isMobile(): boolean {
  return /Mobi/.test(navigator.userAgent) ||
    (window.innerWidth <= 800 && window.innerHeight <= 600) ||
    ('ontouchstart' in window) ||
    (navigator.maxTouchPoints > 0);
}

export function formatTime(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(2)}m`;
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Number.parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`;
}
