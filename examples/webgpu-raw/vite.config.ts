import { defineConfig } from 'vite';

// Mirrors the sibling examples: relative base so the built bundle works under any
// GitHub Pages subpath, and onnxruntime-web is left out of esbuild pre-bundling
// (it ships prebuilt wasm/jsep assets). No models middleware, no weightsUrl —
// the .onnx weights load straight from the default CDN, like a real consumer.
export default defineConfig({
  base: './',
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
