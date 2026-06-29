import { defineConfig } from 'vite';

export default defineConfig({
  // allow serving the .onnx models that live outside this example (repo root)
  server: {
    fs: { allow: ['../..'] },
  },
  // onnxruntime-web ships prebuilt wasm/jsep assets; don't let esbuild pre-bundle it
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
