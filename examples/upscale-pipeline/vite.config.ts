import { defineConfig } from 'vite';

// Loads .onnx weights straight from the default CDN (like the gallery), so this
// builds and runs as a real consumer would. Three + the upscaler are served as
// source (not pre-bundled) and deduped so there is exactly ONE three instance —
// otherwise the upscaler's TempNode subclass wouldn't be recognized by our
// three/webgpu, the same single-instance rule the pathtracer example documents.
export default defineConfig({
  base: './',
  esbuild: { target: 'esnext' }, // three r185 / TSL use top-level await
  build: { target: 'esnext' },
  resolve: {
    dedupe: ['three'],
  },
  server: { fs: { allow: ['../..'] } },
  optimizeDeps: {
    exclude: ['onnxruntime-web', 'three', 'three/webgpu', 'three/tsl', '@pmndrs/upscaler'],
    esbuildOptions: { target: 'esnext' },
  },
});
