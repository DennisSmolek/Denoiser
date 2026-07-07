/**
 * Pack an LDraw model into a single standalone .mpd by inlining every
 * referenced part from gkjohnson/ldraw-parts-library (the CORS-friendly
 * mirror of the official LDraw parts library, same one the
 * three-gpu-pathtracer lego demo resolves against).
 *
 * Port of three.js utils/packLDrawModel.mjs with the filesystem lookups
 * replaced by CDN fetches (plus a local disk cache under .ldraw-cache/).
 *
 * Usage: node tools/pack-ldraw.mjs public/assets/eiffel-tower.mpd public/assets/eiffel-tower_Packed.mpd
 */

/* global process, console, fetch */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const LIB = 'https://raw.githubusercontent.com/gkjohnson/ldraw-parts-library/master/complete/ldraw/';
const CACHE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '.ldraw-cache');

const [, , inFile, outFile] = process.argv;
if (!inFile || !outFile) {
  console.log('Usage: node pack-ldraw.mjs <model.mpd> <out_Packed.mpd>');
  process.exit(1);
}

fs.mkdirSync(CACHE_DIR, { recursive: true });

async function fetchLibraryFile(relPath) {
  const cacheFile = path.join(CACHE_DIR, relPath.replace(/\//g, '__'));
  if (fs.existsSync(cacheFile)) return fs.readFileSync(cacheFile, 'utf8');
  const res = await fetch(LIB + relPath.split('/').map(encodeURIComponent).join('/'));
  if (!res.ok) return null;
  const text = await res.text();
  fs.writeFileSync(cacheFile, text);
  return text;
}

const objectsPaths = [];
const objectsContents = [];
const pathMap = {};
const listOfNotFound = [];

// Same lookup order as packLDrawModel.mjs / LDrawLoader: as-is, then under
// parts/, p/, models/; each tried verbatim then lowercased.
async function locate(fileName) {
  for (const name of [fileName, fileName.toLowerCase()]) {
    let prefixes = ['parts/', 'p/', 'models/', ''];
    if (name.startsWith('48/')) prefixes = ['p/', ...prefixes];
    else if (name.startsWith('s/')) prefixes = ['parts/', ...prefixes];
    for (const prefix of prefixes) {
      const content = await fetchLibraryFile(prefix + name);
      if (content !== null) return { objectPath: (prefix + name).replace(/\\/g, '/'), content };
    }
  }
  return null;
}

async function parseObject(fileName, rootContent = null) {
  const located = rootContent !== null
    ? { objectPath: fileName, content: rootContent }
    : await locate(fileName.trim().replace(/\\/g, '/'));

  if (!located) {
    // Not found in the library — may be an embedded MPD subfile; checked at the end.
    listOfNotFound.push(fileName);
    return null;
  }

  const objectPath = located.objectPath;
  const isRoot = rootContent !== null;
  let content = located.content.replace(/\r\n/g, '\n');
  let processed = isRoot ? '' : `0 FILE ${objectPath}\n`;

  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trimStart();

    if (line.startsWith('0 FILE ')) {
      // Drop the file's own first FILE header. For parts it's re-added above;
      // for the root it MUST go so the main model stays in the top-level
      // document (a leading `0 FILE` would demote it to an unused subfile).
      if (i === 0) continue;
      // Embedded MPD subfile: record its name so refs to it aren't fetched.
      const sub = line.substring('0 FILE '.length).trim().replace(/\\/g, '/');
      if (sub && !pathMap[sub]) pathMap[sub] = sub;
      processed += line + '\n';
      continue;
    }

    if (line.startsWith('1 ')) {
      // type-1 line: "1 <color> x y z  a..i  <file>" — the file is token 14+.
      const m = line.match(/^1(\s+\S+){13}\s+/);
      const ref = m ? line.substring(m[0].length).trim().replace(/\\/g, '/') : '';
      if (ref) {
        if (!(ref in pathMap) || pathMap[ref] === undefined) {
          pathMap[ref] = (await parseObject(ref)) ?? ref;
        }
        processed += line.substring(0, m[0].length) + pathMap[ref] + '\n';
        continue;
      }
    }

    processed += line + '\n';
  }

  if (!objectsPaths.includes(objectPath)) {
    objectsPaths.push(objectPath);
    objectsContents.push(processed);
    console.log(`packed ${objectPath} (${objectsPaths.length})`);
  }
  return objectPath;
}

const materials = await fetchLibraryFile('LDConfig.ldr');
if (!materials) throw new Error('could not fetch LDConfig.ldr');

await parseObject(path.basename(inFile), fs.readFileSync(inFile, 'utf8'));

// Anything still unresolved must exist as an embedded "0 FILE" subfile.
const missing = listOfNotFound.filter((n) => !pathMap[n] || pathMap[n] === n && !objectsPaths.includes(n));
const reallyMissing = missing.filter((n) => !pathMap[n]);
if (reallyMissing.length) {
  console.error('NOT FOUND:', reallyMissing);
  process.exit(1);
}

let packed = materials.replace(/\r\n/g, '\n') + '\n';
for (let i = objectsPaths.length - 1; i >= 0; i--) packed += objectsContents[i];
fs.writeFileSync(outFile, packed + '\n');
console.log(`wrote ${outFile} (${(packed.length / 1e6).toFixed(1)} MB, ${objectsPaths.length} files inlined)`);
