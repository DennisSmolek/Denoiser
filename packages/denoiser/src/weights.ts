// Loads and caches the converted ONNX models (replaces the old TZA weight loader).
// Models are large and hosted on a CDN (like the old tzas/); point `url`/`path`
// at where the .onnx files live. Cached by name + precision.
export class Models {
  private static instance: Models;
  private cache = new Map<string, Uint8Array>();

  /** Subdirectory under the site root (used when `url` is unset). */
  path?: string;
  /** Remote source for the models. */
  url = 'https://cdn.jsdelivr.net/gh/pmndrs/denoiser-weights@models-v1/models';
  /** fp32 (default) or fp16 (smaller/faster, needs the shader-f16 feature). */
  precision: 'fp32' | 'fp16' = 'fp32';

  static getInstance(): Models {
    if (!Models.instance) Models.instance = new Models();
    return Models.instance;
  }

  private fileFor(name: string): string {
    const suffix = this.precision === 'fp16' ? '.fp16.onnx' : '.onnx';
    if (this.url) return `${this.url}/${name}${suffix}`;
    return `/${this.path ?? 'models'}/${name}${suffix}`;
  }

  async get(name: string, overrideUrl?: string): Promise<Uint8Array> {
    const key = `${name}.${this.precision}`;
    const cached = this.cache.get(key);
    if (cached) return cached;
    const url = overrideUrl ?? this.fileFor(name);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Denoiser: failed to load model from ${url} (${res.status})`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    this.cache.set(key, bytes);
    return bytes;
  }

  has(name: string): boolean {
    return this.cache.has(`${name}.${this.precision}`);
  }
}
