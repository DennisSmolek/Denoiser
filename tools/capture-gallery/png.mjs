// Minimal PNG decoder — just enough to independently verify the captured assets
// (dimensions + per-channel/luma pixel stats). Handles 8-bit truecolor (type 2)
// and truecolor+alpha (type 6), which is what canvas.toDataURL('image/png')
// produces. No deps beyond node:zlib.
import zlib from 'node:zlib';

const SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/** Decode a PNG buffer -> { width, height, channels, data: Uint8Array RGBA }. */
export function decodePng(buf) {
  if (!buf.subarray(0, 8).equals(SIG)) throw new Error('not a PNG (bad signature)');
  let off = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off);
    const type = buf.toString('ascii', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    off += 12 + len;
  }
  if (bitDepth !== 8) throw new Error(`unsupported bit depth ${bitDepth}`);
  const channels = colorType === 6 ? 4 : colorType === 2 ? 3 : 0;
  if (!channels) throw new Error(`unsupported color type ${colorType}`);

  const raw = zlib.inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = new Uint8Array(width * height * 4);
  const cur = new Uint8Array(stride);
  const prev = new Uint8Array(stride);
  let p = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[p++];
    for (let x = 0; x < stride; x++) {
      const rawByte = raw[p++];
      const a = x >= channels ? cur[x - channels] : 0;
      const b = prev[x];
      const c = x >= channels ? prev[x - channels] : 0;
      let val;
      switch (filter) {
        case 0: val = rawByte; break;
        case 1: val = rawByte + a; break;
        case 2: val = rawByte + b; break;
        case 3: val = rawByte + ((a + b) >> 1); break;
        case 4: val = rawByte + paeth(a, b, c); break;
        default: throw new Error(`unknown filter ${filter}`);
      }
      cur[x] = val & 0xff;
    }
    // expand to RGBA
    for (let x = 0; x < width; x++) {
      const di = (y * width + x) * 4;
      const si = x * channels;
      out[di] = cur[si];
      out[di + 1] = cur[si + 1];
      out[di + 2] = cur[si + 2];
      out[di + 3] = channels === 4 ? cur[si + 3] : 255;
    }
    prev.set(cur);
  }
  return { width, height, channels, data: out };
}

/** Pixel stats from decoded RGBA — independent of the in-browser numbers. */
export function pngStats(buf) {
  const { width: w, height: h, data } = decodePng(buf);
  let sum = 0;
  let min = 255;
  let max = 0;
  let nonBlack = 0;
  const luma = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const l = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    luma[i] = l;
    sum += l;
    if (l < min) min = l;
    if (l > max) max = l;
    if (l > 4) nonBlack++;
  }
  let vacc = 0;
  let vn = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let s = 0;
      let s2 = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const l = luma[(y + dy) * w + (x + dx)];
          s += l;
          s2 += l * l;
        }
      }
      const m = s / 9;
      vacc += s2 / 9 - m * m;
      vn++;
    }
  }
  return {
    width: w,
    height: h,
    meanLuma: +(sum / (w * h)).toFixed(2),
    minLuma: +min.toFixed(1),
    maxLuma: +max.toFixed(1),
    nonBlackFraction: +(nonBlack / (w * h)).toFixed(4),
    localVariance: +(vacc / vn).toFixed(2),
  };
}
