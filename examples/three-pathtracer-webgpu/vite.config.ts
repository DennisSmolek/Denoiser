import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { createReadStream, createWriteStream, existsSync, statSync, mkdirSync } from 'node:fs';
import path from 'node:path';

const modelsDir = fileURLToPath(new URL('../../packages/denoiser/models', import.meta.url));
// Split-graph workaround artifacts (<name>.tail.onnx / <name>.enc0.bin), served
// under /models alongside the real models. Checked first.
const splitDir = fileURLToPath(new URL('./split-models', import.meta.url));

export default defineConfig({
  server: { fs: { allow: ['../..'] } },
  // three r185 / pathtracer source use top-level await -> need an esnext target
  esbuild: { target: 'esnext' },
  build: { target: 'esnext' },
  // Exactly one three instance: the pathtracer (served as source) and our
  // `three/webgpu` import must resolve to the same copy, or material nodes from
  // one instance are unrecognized by the other (-> "bsdfSample of null").
  // Use the worktree's denoiser SOURCE (not the built dist) so local engine edits
  // — e.g. the aux split-graph workaround — are picked up without a rebuild.
  resolve: {
    dedupe: ['three', 'three-mesh-bvh'],
    alias: { denoiser: fileURLToPath(new URL('../../packages/denoiser/src/index.ts', import.meta.url)) },
  },
  optimizeDeps: {
    // Serve three + the pathtracer + mesh-bvh as source (not pre-bundled): keeps a
    // single three instance and lets the pathtracer's import.meta.url asset resolve.
    exclude: [
      'onnxruntime-web',
      'three',
      'three/webgpu',
      'three/tsl',
      'three-mesh-bvh',
      'three-gpu-pathtracer',
    ],
    esbuildOptions: { target: 'esnext' },
  },
  plugins: [
    {
      // Debug: accept raw binary dumps from the page (POST /dump/<name>) and
      // write them under ./dumps — used by the native-OIDN reference harness
      // (tools/oidn-native-compare) to capture the exact float inputs.
      name: 'dump-endpoint',
      configureServer(server) {
        const dumpsDir = fileURLToPath(new URL('./dumps', import.meta.url));
        server.middlewares.use('/dump', (req, res, next) => {
          if (req.method !== 'POST') return next();
          mkdirSync(dumpsDir, { recursive: true });
          const name = path.basename(decodeURIComponent((req.url ?? '/x').slice(1).split('?')[0] || 'dump.bin'));
          const out = createWriteStream(path.join(dumpsDir, name));
          req.pipe(out).on('finish', () => { res.statusCode = 200; res.end('ok'); });
        });
      },
    },
    {
      name: 'serve-models',
      configureServer(server) {
        server.middlewares.use('/models', (req, res, next) => {
          const rel = decodeURIComponent((req.url ?? '').split('?')[0]);
          const split = path.join(splitDir, rel);
          const file = (existsSync(split) && statSync(split).isFile())
            ? split
            : path.join(modelsDir, rel);
          if (!existsSync(file) || !statSync(file).isFile()) return next();
          res.setHeader('Content-Type', 'application/octet-stream');
          createReadStream(file).pipe(res);
        });
      },
    },
  ],
});
