#!/usr/bin/env node
// Example launcher / index.
//   node tools/examples.mjs            -> print the index of runnable examples
//   node tools/examples.mjs <name>     -> `yarn workspace <name> dev`
// Names may be abbreviated (any unique prefix / substring), e.g. `three`, `test`, `bench`, `smoke`.
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import path from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const examplesDir = path.join(root, 'examples');

const examples = readdirSync(examplesDir, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => path.join(examplesDir, d.name, 'package.json'))
  .filter(existsSync)
  .map((pkgPath) => {
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8'));
    const dev = pkg.scripts?.dev ?? '';
    const port = dev.match(/--port\s+(\d+)/)?.[1] ?? '5173';
    return { name: pkg.name, port, description: pkg.description ?? '', hasDev: Boolean(pkg.scripts?.dev) };
  })
  .filter((e) => e.hasDev)
  .sort((a, b) => a.port.localeCompare(b.port));

const query = process.argv[2];

if (!query) {
  const pad = Math.max(...examples.map((e) => e.name.length));
  console.log('\nRunnable examples (yarn dev:<name> or yarn examples <name>).');
  console.log('Ports are preferred; Vite falls through to the next free one if taken.\n');
  for (const e of examples) {
    console.log(`  ${e.name.padEnd(pad)}  :${e.port}+  ${e.description}`);
  }
  console.log('');
  process.exit(0);
}

const matches = examples.filter((e) => e.name === query)
  .concat(examples.filter((e) => e.name !== query && e.name.includes(query)));

if (matches.length === 0) {
  console.error(`No example matching "${query}". Run \`yarn examples\` to see the list.`);
  process.exit(1);
}
if (matches.length > 1 && matches[0].name !== query) {
  console.error(`"${query}" is ambiguous: ${matches.map((m) => m.name).join(', ')}`);
  process.exit(1);
}

const target = matches[0];
console.log(`Starting ${target.name} (preferred port ${target.port}, next free if taken) ...\n`);
spawn('yarn', ['workspace', target.name, 'dev'], { stdio: 'inherit', cwd: root, shell: process.platform === 'win32' })
  .on('exit', (code) => process.exit(code ?? 0));
