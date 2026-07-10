import { defineConfig } from 'vite';

// No models middleware, no weightsUrl override: loads the .onnx weights straight
// from the default CDN, same as a real consumer would.
export default defineConfig({
  // Relative base so the built bundle works under any subpath (GitHub Pages
  // serves this at /denoiser/aux-inputs/). Single-page app, no absolute asset refs.
  base: './',
  // onnxruntime-web ships prebuilt wasm/jsep assets; don't let esbuild pre-bundle it
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
