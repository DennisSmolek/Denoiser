# denoiser-react (deprecated)

This v1 React wrapper is **retired**. `denoiser` 2.x has a stateless, two-call
API (`Denoiser.create()` once, then `denoise()` per frame) that doesn't need a
wrapper — a plain `useEffect`/ref pattern is enough:

```tsx
import { useEffect, useRef } from 'react';
import { Denoiser } from 'denoiser';

function DenoisedCanvas({ noisyImage }: { noisyImage: HTMLImageElement }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const denoiserRef = useRef<Denoiser>();

  useEffect(() => {
    let cancelled = false;
    Denoiser.create({ precision: 'fp16' }).then((d) => {
      if (cancelled) return;
      denoiserRef.current = d;
    });
    return () => { cancelled = true; denoiserRef.current?.destroyDevice(); };
  }, []);

  useEffect(() => {
    denoiserRef.current?.denoise(noisyImage).then((clean) => {
      canvasRef.current?.getContext('2d')?.putImageData(clean, 0, 0);
    });
  }, [noisyImage]);

  return <canvas ref={canvasRef} />;
}
```

See [`docs/guides/migrating-from-v1.md`](../../docs/guides/migrating-from-v1.md)
for the full v1 → v2 API mapping.

The v1 wrapper's source is still in this package's `src/` for reference and
remains available on the 0.x release line; it is no longer built or published
from this repo.
