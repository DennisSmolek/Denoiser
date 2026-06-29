import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { createReadStream, existsSync, statSync } from 'node:fs';
import path from 'node:path';

const modelsDir = fileURLToPath(new URL('../../packages/denoiser/models', import.meta.url));

export default defineConfig({
  server: { fs: { allow: ['../..'] } },
  // three r185 / pathtracer source use top-level await -> need an esnext target
  esbuild: { target: 'esnext' },
  build: { target: 'esnext' },
  optimizeDeps: { exclude: ['onnxruntime-web'], esbuildOptions: { target: 'esnext' } },
  plugins: [
    {
      name: 'serve-models',
      configureServer(server) {
        server.middlewares.use('/models', (req, res, next) => {
          const file = path.join(modelsDir, decodeURIComponent((req.url ?? '').split('?')[0]));
          if (!existsSync(file) || !statSync(file).isFile()) return next();
          res.setHeader('Content-Type', 'application/octet-stream');
          createReadStream(file).pipe(res);
        });
      },
    },
  ],
});
