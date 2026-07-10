// Shared "demo chrome" for the denoiser examples — dependency-free.
//
// Imported via a RELATIVE path (e.g. `import { ensureWebGPU } from '../../_shared/chrome'`),
// NOT as a workspace package. Everything here is plain DOM + inline styles so any
// example (three-based or not, bundled or raw) can use it without pulling deps.
//
// Public API:
//   ensureWebGPU(el?)  -> Promise<boolean>   feature-detect + friendly banner on failure
//   statsOverlay()     -> { frame(ms) }      fixed-corner ms/FPS readout
//   demoFooter(name)   -> void               small footer (name, source link, back to index)
//   pathtracerNote()   -> void               dismissible note re: the unreleased pathtracer branch

const REPO_URL = 'https://github.com/pmndrs/denoiser';
const SUPPORTED_BROWSERS = 'Chrome 113+, Edge 113+, or Safari 26+ (Technology Preview)';
const PATHTRACER_URL = 'https://github.com/gkjohnson/three-gpu-pathtracer/tree/webgpu-pathtracer';

/**
 * Feature-detect WebGPU (navigator.gpu + a real adapter). On failure, injects a
 * friendly full-page banner explaining what's needed and returns `false`; on
 * success returns `true` and injects nothing.
 *
 * @param el Optional container for the banner (defaults to <body>).
 */
export async function ensureWebGPU(el?: HTMLElement): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  let ok = false;
  let detail = '';
  if (!gpu) {
    detail = 'navigator.gpu is undefined — this browser has no WebGPU.';
  } else {
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter) ok = true;
      else detail = 'navigator.gpu is present but no GPU adapter is available (try enabling hardware acceleration).';
    } catch (err) {
      detail = `requestAdapter() failed: ${(err as Error).message}`;
    }
  }
  if (ok) return true;
  injectUnsupportedBanner(el ?? document.body, detail);
  return false;
}

function injectUnsupportedBanner(host: HTMLElement, detail: string): void {
  const banner = document.createElement('div');
  banner.setAttribute('data-denoiser-chrome', 'unsupported');
  banner.style.cssText = [
    'position:fixed', 'inset:0', 'z-index:99999',
    'display:flex', 'align-items:center', 'justify-content:center',
    'padding:2rem', 'box-sizing:border-box',
    'background:#0b0e14', 'color:#e2e8f0',
    'font-family:system-ui,-apple-system,sans-serif', 'line-height:1.5',
  ].join(';');
  banner.innerHTML = `
    <div style="max-width:34rem;text-align:center">
      <div style="font-size:2.5rem;margin-bottom:0.5rem">⚡</div>
      <h1 style="margin:0 0 0.5rem;font-size:1.4rem">This demo needs WebGPU</h1>
      <p style="margin:0 0 1rem;color:#a0aec0">
        The denoiser runs entirely on the GPU via WebGPU, which isn't available here.
        Try a supported browser: ${SUPPORTED_BROWSERS}.
      </p>
      <p style="margin:0 0 1.5rem;color:#718096;font-size:0.85rem">${escapeHtml(detail)}</p>
      <a href="${REPO_URL}" style="color:#90cdf4">denoiser on GitHub →</a>
    </div>`;
  host.appendChild(banner);
}

/**
 * Tiny fixed-corner ms/FPS readout. Call `frame(ms)` once per rendered frame
 * (or per denoise) with the elapsed milliseconds; the overlay shows a smoothed
 * ms + FPS. No deps, no rAF of its own.
 */
export function statsOverlay(): { frame(ms: number): void } {
  const el = document.createElement('div');
  el.setAttribute('data-denoiser-chrome', 'stats');
  el.style.cssText = [
    'position:fixed', 'top:8px', 'right:8px', 'z-index:99998',
    'padding:4px 8px', 'border-radius:6px',
    'background:rgba(11,14,20,0.72)', 'color:#7ee787',
    'font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace',
    'pointer-events:none', 'white-space:pre', 'letter-spacing:0.02em',
  ].join(';');
  el.textContent = '-- ms';
  document.body.appendChild(el);

  let smoothed = 0;
  return {
    frame(ms: number) {
      // Exponential moving average so the readout doesn't jitter.
      smoothed = smoothed === 0 ? ms : smoothed * 0.9 + ms * 0.1;
      const fps = smoothed > 0 ? 1000 / smoothed : 0;
      el.textContent = `${smoothed.toFixed(1)} ms\n${fps.toFixed(0)} fps`;
    },
  };
}

/**
 * Injects a small footer: the demo name, a link to this demo's source on GitHub,
 * and a link back to the demo index (`../`). Safe to call once per page.
 */
export function demoFooter(name: string): void {
  const footer = document.createElement('footer');
  footer.setAttribute('data-denoiser-chrome', 'footer');
  footer.style.cssText = [
    'margin-top:3rem', 'padding:1rem 0', 'border-top:1px solid rgba(128,128,128,0.25)',
    'color:#718096', 'font:0.85rem/1.5 system-ui,-apple-system,sans-serif',
    'display:flex', 'gap:1rem', 'flex-wrap:wrap', 'align-items:center',
  ].join(';');
  const sourceUrl = `${REPO_URL}/tree/main/examples/${name}`;
  footer.innerHTML = `
    <span>denoiser · <strong>${escapeHtml(name)}</strong></span>
    <a href="${sourceUrl}" style="color:#90cdf4">source ↗</a>
    <a href="../" style="color:#90cdf4">← all demos</a>`;
  document.body.appendChild(footer);
}

/**
 * Small dismissible note (call near `demoFooter`, e.g. right after it): the
 * WebGPU path tracer these demos drive is gkjohnson's unreleased
 * `webgpu-pathtracer` branch of three-gpu-pathtracer, SHA-pinned in
 * package.json (not a released version) — links to the branch for context.
 * Dismissal is remembered in localStorage so it doesn't nag on every visit.
 */
export function pathtracerNote(): void {
  const KEY = 'denoiser-chrome:pathtracer-note-dismissed';
  try {
    if (localStorage.getItem(KEY) === '1') return;
  } catch {
    /* localStorage unavailable (private mode etc.) — just show the note every time */
  }
  const note = document.createElement('div');
  note.setAttribute('data-denoiser-chrome', 'pathtracer-note');
  note.style.cssText = [
    'margin-top:1rem', 'padding:0.6rem 0.9rem', 'border-radius:8px',
    'background:rgba(144,205,244,0.08)', 'border:1px solid rgba(144,205,244,0.25)',
    'color:#a0aec0', 'font:0.8rem/1.5 system-ui,-apple-system,sans-serif',
    'display:flex', 'gap:0.75rem', 'align-items:flex-start', 'justify-content:space-between',
  ].join(';');
  note.innerHTML = `
    <span>Path tracing here runs on gkjohnson's unreleased
      <a href="${PATHTRACER_URL}" target="_blank" rel="noopener" style="color:#90cdf4">three-gpu-pathtracer
      <code>webgpu-pathtracer</code> branch</a>, pinned to a fixed commit — expect rough edges.</span>
    <button type="button" aria-label="dismiss" style="flex:none;background:none;border:none;color:#718096;cursor:pointer;font-size:1rem;line-height:1;padding:0 0.2rem">&times;</button>`;
  note.querySelector('button')!.addEventListener('click', () => {
    try { localStorage.setItem(KEY, '1'); } catch { /* ignore */ }
    note.remove();
  });
  document.body.appendChild(note);
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
