#!/usr/bin/env node
// Gallery-asset capture (Phase B B1.3).
//
// Drives a pathtracer example in headless Chrome (WebGPU) and writes, per scene,
// the spp ladder + reference + albedo/normal AOVs the gallery demo consumes into
// examples/gallery/public/scenes/<id>/, then merges public/scenes/manifest.json.
//
// Usage:
//   node tools/capture-gallery/capture.mjs [sceneId ...] [--ref N] [--attempts N] [--headed]
//   node tools/capture-gallery/capture.mjs            # all scenes
//   node tools/capture-gallery/capture.mjs eiffel --ref 128
//
// No puppeteer: spawns the example's vite dev server + headless Chrome, talks CDP
// over Node's native WebSocket. Each scene is retried (default 5x) with a hard
// per-attempt timeout — the pathtracer branch hangs nondeterministically during
// headless init. If a scene never inits headless, fall back to the ?capture=1
// button in a real browser (see README).
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync, readFileSync, openSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';
import { pngStats } from './png.mjs';

const ROOT = fileURLToPath(new URL('../..', import.meta.url));
const OUT_ROOT = path.join(ROOT, 'examples/gallery/public/scenes');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const VITE_BIN = path.join(ROOT, 'node_modules/vite/bin/vite.js');

// Per-scene config. `ready` is a substring that appears in #status once the path
// tracer has initialized (i.e. __captureGallery is safe to call).
const SCENES = {
  spheres: {
    dir: path.join(ROOT, 'examples/three-pathtracer-webgpu'),
    port: 5173,
    ready: 'path tracer initialized',
    referenceSpp: 256,
  },
  eiffel: {
    dir: path.join(ROOT, 'examples/ldraw-eiffel'),
    port: 5177,
    ready: 'path tracer BVH built',
    referenceSpp: 256,
  },
};

// --- CLI ---
const argv = process.argv.slice(2);
const flags = { attempts: 5, ref: null, headed: false };
const wanted = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--attempts') flags.attempts = parseInt(argv[++i], 10);
  else if (a === '--ref') flags.ref = parseInt(argv[++i], 10);
  else if (a === '--headed') flags.headed = true;
  else if (a.startsWith('--')) throw new Error(`unknown flag ${a}`);
  else wanted.push(a);
}
const scenes = wanted.length ? wanted : Object.keys(SCENES);
for (const s of scenes) if (!SCENES[s]) throw new Error(`unknown scene "${s}" (known: ${Object.keys(SCENES).join(', ')})`);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);

async function waitServer(port, timeoutMs) {
  // Probe via fetch on `localhost` (not a raw 127.0.0.1 socket): vite v8 may bind
  // IPv6 ::1 only, which a 127.0.0.1 connect would miss.
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://localhost:${port}/`);
      if (r.ok || r.status === 200) return true;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  return false;
}

async function httpJson(url, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  let lastErr;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      return await r.json();
    } catch (e) {
      lastErr = e;
      await sleep(250);
    }
  }
  throw lastErr ?? new Error(`httpJson timeout ${url}`);
}

// Minimal CDP client over a single page target's WebSocket.
class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async evalJs(expression, awaitPromise = false) {
    const r = await this.send('Runtime.evaluate', {
      expression, awaitPromise, returnByValue: true,
    });
    if (r.exceptionDetails) {
      throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    }
    return r.result.value;
  }
}

function connectWs(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.addEventListener('open', () => resolve(ws));
    ws.addEventListener('error', (e) => reject(new Error('ws error: ' + (e.message ?? 'unknown'))));
  });
}

function dataUrlToBuffer(dataUrl) {
  const comma = dataUrl.indexOf(',');
  return Buffer.from(dataUrl.slice(comma + 1), 'base64');
}

// One capture attempt against an already-running dev server. Launches a fresh
// Chrome, drives the page, writes the PNGs. Throws on any failure/timeout.
async function attemptCapture(sceneId, cfg, attemptTimeoutMs) {
  const debugPort = 9222 + Math.floor(Math.random() * 500);
  const userDataDir = path.join(os.tmpdir(), `capture-gallery-${sceneId}-${Date.now()}`);
  const url = `http://localhost:${cfg.port}/?headless=1`;
  const chromeArgs = [
    flags.headed ? '--headless=new' : '--headless=new',
    '--no-sandbox', '--use-angle=metal', '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${userDataDir}`,
    '--window-size=640,640',
    url,
  ];
  const chrome = spawn(CHROME, chromeArgs, { stdio: 'ignore' });
  let ws;
  const killChrome = () => { try { chrome.kill('SIGKILL'); } catch { /* noop */ } };

  try {
    return await withTimeout(attemptTimeoutMs, async () => {
      // Find the page target.
      const targets = await httpJson(`http://127.0.0.1:${debugPort}/json`, 15000);
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (!page) throw new Error('no page target');
      ws = await connectWs(page.webSocketDebuggerUrl);
      const cdp = new CDP(ws);
      await cdp.send('Runtime.enable');
      await cdp.send('Page.enable');

      // Poll #status for the ready signal (init can hang here — that's the risk).
      let ready = false;
      const readyDeadline = Date.now() + Math.min(attemptTimeoutMs - 5000, 90000);
      let lastStatus = '';
      while (Date.now() < readyDeadline) {
        const status = await cdp.evalJs(
          `(document.querySelector('#status')?.textContent) || ''`,
        ).catch(() => '');
        lastStatus = status;
        if (status.includes('ERROR')) throw new Error('page error: ' + status.split('\n').find((l) => l.includes('ERROR')));
        if (status.includes(cfg.ready)) { ready = true; break; }
        await sleep(500);
      }
      if (!ready) throw new Error(`init did not reach "${cfg.ready}" (last status: ${lastStatus.slice(-160).replace(/\n/g, ' | ')})`);
      const refSpp = flags.ref ?? cfg.referenceSpp;
      log(`  ${sceneId}: initialized; running capture (ref ${refSpp} spp)…`);

      // Run the capture. It stores the full result on window.__captureResult;
      // return only the small metadata+stats blob here (data URLs fetched below).
      const meta = await cdp.evalJs(
        `(async () => { const r = await window.__captureGallery({ referenceSpp: ${refSpp} });
          return JSON.stringify({ id:r.id, title:r.title, width:r.width, height:r.height,
            spp:r.spp, referenceSpp:r.referenceSpp, stats:r.stats }); })()`,
        true,
      );
      const m = JSON.parse(meta);

      // Pull each asset's data URL individually (keeps CDP responses small).
      const outDir = path.join(OUT_ROOT, sceneId);
      mkdirSync(outDir, { recursive: true });
      const files = {};
      const fetchAsset = async (jsExpr, filename) => {
        const dataUrl = await cdp.evalJs(jsExpr);
        writeFileSync(path.join(outDir, filename), dataUrlToBuffer(dataUrl));
        files[filename] = pngStats(readFileSync(path.join(outDir, filename)));
      };
      for (const spp of m.spp) {
        await fetchAsset(`window.__captureResult.color['${spp}']`, `spp${spp}.png`);
      }
      await fetchAsset(`window.__captureResult.reference`, 'reference.png');
      await fetchAsset(`window.__captureResult.albedo`, 'albedo.png');
      await fetchAsset(`window.__captureResult.normal`, 'normal.png');

      return { meta: m, files };
    });
  } finally {
    try { ws?.close(); } catch { /* noop */ }
    killChrome();
  }
}

function withTimeout(ms, fn) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`attempt timeout after ${ms}ms`)), ms);
    fn().then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
  });
}

async function captureScene(sceneId) {
  const cfg = SCENES[sceneId];
  log(`=== scene "${sceneId}" — starting dev server (port ${cfg.port}) ===`);
  const viteLog = path.join(os.tmpdir(), `capture-gallery-vite-${sceneId}.log`);
  const viteOut = openSync(viteLog, 'w');
  const vite = spawn(process.execPath, [VITE_BIN, '--port', String(cfg.port), '--strictPort'], {
    cwd: cfg.dir, stdio: ['ignore', viteOut, viteOut], env: { ...process.env },
  });
  try {
    // Cold start prebundles three + the pathtracer via esbuild — can take a
    // while the first time (warm cache is ~8s). Give it room.
    if (!(await waitServer(cfg.port, 120000))) {
      throw new Error(`dev server did not start (see ${viteLog})`);
    }
    await sleep(1000);
    for (let attempt = 1; attempt <= flags.attempts; attempt++) {
      log(`--- ${sceneId}: attempt ${attempt}/${flags.attempts} ---`);
      try {
        const res = await attemptCapture(sceneId, cfg, 240000);
        log(`  ${sceneId}: capture OK`);
        return res;
      } catch (e) {
        log(`  ${sceneId}: attempt ${attempt} failed: ${e.message}`);
        if (attempt < flags.attempts) await sleep(1500);
      }
    }
    throw new Error(`all ${flags.attempts} attempts failed`);
  } finally {
    try { vite.kill('SIGKILL'); } catch { /* noop */ }
  }
}

function mergeManifest(entries) {
  const manifestPath = path.join(OUT_ROOT, 'manifest.json');
  let manifest = { scenes: [] };
  if (existsSync(manifestPath)) {
    try { manifest = JSON.parse(readFileSync(manifestPath, 'utf8')); } catch { /* start fresh */ }
    if (!Array.isArray(manifest.scenes)) manifest.scenes = [];
  }
  for (const m of entries) {
    const entry = {
      id: m.id,
      title: m.title,
      width: m.width,
      height: m.height,
      spp: m.spp,
      color: Object.fromEntries(m.spp.map((s) => [String(s), `spp${s}.png`])),
      albedo: 'albedo.png',
      normal: 'normal.png',
      reference: 'reference.png',
    };
    const idx = manifest.scenes.findIndex((x) => x.id === m.id);
    if (idx >= 0) manifest.scenes[idx] = entry;
    else manifest.scenes.push(entry);
  }
  mkdirSync(OUT_ROOT, { recursive: true });
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  return manifestPath;
}

async function main() {
  log(`capturing scenes: ${scenes.join(', ')} (attempts=${flags.attempts}${flags.ref ? `, ref=${flags.ref}` : ''})`);
  const captured = [];
  const failures = [];
  for (const sceneId of scenes) {
    try {
      const res = await captureScene(sceneId);
      captured.push(res);
    } catch (e) {
      log(`!!! scene "${sceneId}" FAILED: ${e.message}`);
      failures.push(sceneId);
    }
  }

  if (captured.length) {
    const manifestPath = mergeManifest(captured.map((c) => c.meta));
    log(`\n=== manifest merged: ${manifestPath} ===`);
    // Report per-file independent PNG stats.
    for (const c of captured) {
      log(`\nscene "${c.meta.id}" (${c.meta.title}) — ref ${c.meta.referenceSpp} spp:`);
      for (const [file, st] of Object.entries(c.files)) {
        log(`  ${file.padEnd(14)} ${st.width}x${st.height}  meanLuma=${String(st.meanLuma).padStart(6)}  ` +
          `nonBlack=${(st.nonBlackFraction * 100).toFixed(1).padStart(5)}%  var=${String(st.localVariance).padStart(7)}`);
      }
    }
  }
  if (failures.length) {
    log(`\n!!! ${failures.length} scene(s) failed headless: ${failures.join(', ')}`);
    log('    Fallback: run `<scene> dev` and open http://localhost:<port>/?capture=1 in Chrome,');
    log('    click "capture gallery assets", and save the downloads into public/scenes/<id>/.');
    log('    See tools/capture-gallery/README.md.');
    process.exitCode = 1;
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
