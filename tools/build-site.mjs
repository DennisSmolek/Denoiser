#!/usr/bin/env node
// Build the deployed examples and assemble the exact static tree that ships to
// GitHub Pages. Used by BOTH `.github/workflows/pages.yml` and local verification
// so the two can never drift.
//
// Layout produced (repo-root `_site/`):
//   _site/index.html            <- examples/site/ (the landing page)
//   _site/<example>/...         <- examples/<example>/dist/*
//
// Prereq: `yarn workspace denoiser build` must have run first (the non-aliased
// examples import the built `denoiser` dist). The workflow does that before us.
//
// Usage:
//   node tools/build-site.mjs            build all examples, then assemble
//   node tools/build-site.mjs --no-build assemble from existing dist/ only

import { execSync } from 'node:child_process';
import { cpSync, rmSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const outDir = path.join(repoRoot, '_site');

// Only these examples deploy. Internal harnesses (webgpu-ort-smoke,
// denoiser-package-test, aux-split-verify) are intentionally excluded.
const EXAMPLES = ['hello-world', 'gallery', 'aux-inputs', 'webgpu-raw', 'babylon', 'realtime-compare', 'upscale-pipeline', 'bench', 'ldraw-eiffel', 'three-pathtracer-webgpu'];

const noBuild = process.argv.includes('--no-build');

function run(cmd) {
  console.log(`\n$ ${cmd}`);
  execSync(cmd, { cwd: repoRoot, stdio: 'inherit' });
}

function dirSize(dir) {
  let total = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    total += entry.isDirectory() ? dirSize(p) : statSync(p).size;
  }
  return total;
}

const mb = (bytes) => `${(bytes / 1024 / 1024).toFixed(2)} MB`;

// 1) Build each example.
if (!noBuild) {
  for (const name of EXAMPLES) run(`yarn workspace ${name} build`);
}

// 2) Fresh output tree.
rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

// 3) Landing page at the site root.
cpSync(path.join(repoRoot, 'examples', 'site'), outDir, { recursive: true });

// 4) Each example under its own subpath.
const sizes = [];
for (const name of EXAMPLES) {
  const dist = path.join(repoRoot, 'examples', name, 'dist');
  if (!existsSync(dist)) {
    throw new Error(`Missing build output: ${dist} — run without --no-build first.`);
  }
  cpSync(dist, path.join(outDir, name), { recursive: true });
  sizes.push([name, dirSize(dist)]);
}

// 5) Report.
console.log('\nAssembled _site/:');
console.log(`  index.html (landing page)`);
let total = 0;
for (const [name, bytes] of sizes) {
  total += bytes;
  console.log(`  ${name}/  ${mb(bytes)}`);
}
console.log(`  total examples: ${mb(total)}`);
console.log(`\nDone -> ${outDir}`);
