import { defineConfig } from 'vite';

// No models middleware, no weightsUrl override anywhere in this example: it loads
// the .onnx weights straight from the default CDN, same as a real consumer would.
export default defineConfig({
  // onnxruntime-web ships prebuilt wasm/jsep assets; don't let esbuild pre-bundle it
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
