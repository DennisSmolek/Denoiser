import { defineConfig } from 'vite';

// Loads the .onnx weights straight from the default CDN (like a real consumer);
// no models middleware. Mirrors the three-based examples for the single-three
// instance requirement: three/webgpu + three/tsl + three/addons must all resolve
// to ONE copy, or node classes from one instance are unrecognized by the other.
export default defineConfig({
  // Relative base so the built bundle works under any subpath (GitHub Pages
  // serves this at /denoiser/realtime-compare/).
  base: './',
  server: { fs: { allow: ['../..'] } },
  // three r185 addons use top-level await -> need an esnext target.
  esbuild: { target: 'esnext' },
  build: { target: 'esnext' },
  resolve: {
    dedupe: ['three'],
  },
  optimizeDeps: {
    // Serve three as source (not pre-bundled) to keep a single three instance;
    // onnxruntime-web ships prebuilt wasm/jsep assets — don't pre-bundle it.
    exclude: ['onnxruntime-web', 'three', 'three/webgpu', 'three/tsl'],
    esbuildOptions: { target: 'esnext' },
  },
});
