// WGSL compute pre/post-processing on the shared GPUDevice — the GPU replacement
// for the old TensorFlow.js tensor math (normalization, layout, tiling, blend).
//
// Kernels:
//   extractTiles — pull a BATCH of square tiles (per-tile offsets, zero-padded)
//                  from up to three full RGBA8 images (color [+albedo +normal])
//                  into a planar [B,C,tile,tile] NCHW input in one dispatch
//                  (workgroup z = batch slot). color/albedo normalized to [0,1]
//                  (optionally sRGB->linear); normal mapped to OIDN's [-1,1].
//   accumulate   — blend ONE model-output tile (by batch slot) into accum +
//                  weight buffers using the min-of-sigmoid overlap mask
//                  (matches the old tiler.ts). Overlapping tiles must land in
//                  separate compute passes: pass boundaries synchronize the
//                  read-modify-write on accum/weight; z-batching them would race.
//   resolve      — accum / weight -> RGBA8, optional linear->sRGB, LDR clamp,
//                  optional Y flip.
//
// Layout is NCHW; channels is 3 (color), 6 (+albedo) or 9 (+albedo+normal).

// `io` is the model IO element type: 'f32', or 'f16' for fp16 models (needs the
// shader-f16 device feature). Only the model-facing NCHW buffers change type;
// accum/weight/resolve stay f32.
const EXTRACT_TILES = (io: string) => /* wgsl */ `
${io === 'f16' ? 'enable f16;' : ''}
alias IOType = ${io};
struct P {
  imgW:u32, imgH:u32, tileW:u32, tileH:u32, channels:u32, srgb:u32, count:u32, _p0:u32,
};
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<storage, read> albedo: array<u32>;
@group(0) @binding(2) var<storage, read> normal: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<IOType>; // NCHW, count*channels*tileW*tileH
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> offsets: array<vec2<u32>>; // per-slot startX,startY

fn srgbToLinear(c: vec3<f32>) -> vec3<f32> {
  let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
  let lo = c / 12.92;
  return select(lo, hi, c > vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.z >= p.count || gid.x >= p.tileW || gid.y >= p.tileH) { return; }
  let plane = p.tileW * p.tileH;
  let base = gid.z * p.channels * plane;
  let didx = gid.y * p.tileW + gid.x;
  let off = offsets[gid.z];
  let sx = off.x + gid.x;
  let sy = off.y + gid.y;
  let inside = sx < p.imgW && sy < p.imgH;
  let sidx = select(0u, sy * p.imgW + sx, inside);

  var col = vec3<f32>(0.0);
  if (inside) {
    col = unpack4x8unorm(color[sidx]).xyz;
    if (p.srgb == 1u) { col = srgbToLinear(col); }
  }
  dst[base + 0u * plane + didx] = IOType(col.x);
  dst[base + 1u * plane + didx] = IOType(col.y);
  dst[base + 2u * plane + didx] = IOType(col.z);

  if (p.channels >= 6u) {
    var alb = vec3<f32>(0.0);
    if (inside) { alb = unpack4x8unorm(albedo[sidx]).xyz; }
    dst[base + 3u * plane + didx] = IOType(alb.x);
    dst[base + 4u * plane + didx] = IOType(alb.y);
    dst[base + 5u * plane + didx] = IOType(alb.z);
  }
  if (p.channels >= 9u) {
    var nrm = vec3<f32>(0.0);
    if (inside) { nrm = unpack4x8unorm(normal[sidx]).xyz * 2.0 - 1.0; }
    dst[base + 6u * plane + didx] = IOType(nrm.x);
    dst[base + 7u * plane + didx] = IOType(nrm.y);
    dst[base + 8u * plane + didx] = IOType(nrm.z);
  }
}
`;

const ACCUMULATE_TILE = (io: string) => /* wgsl */ `
${io === 'f16' ? 'enable f16;' : ''}
alias IOType = ${io};
struct P {
  imgW:u32, imgH:u32, startX:u32, startY:u32, curW:u32, curH:u32,
  tileX:u32, tileY:u32, tilesX:u32, tilesY:u32, tileW:u32, tileH:u32,
  batchIdx:u32, overlap:f32,
};
@group(0) @binding(0) var<storage, read> src: array<IOType>;       // NCHW model output (B*3ch)
@group(0) @binding(1) var<storage, read_write> accum: array<f32>;  // 3*imgW*imgH
@group(0) @binding(2) var<storage, read_write> weight: array<f32>; // imgW*imgH
@group(0) @binding(3) var<uniform> p: P;

fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-12.0 * (x - 0.5))); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.curW || gid.y >= p.curH) { return; }
  let tx = gid.x; let ty = gid.y;
  var yW = 1.0; var xW = 1.0;
  if (p.tileY > 0u)            { yW = min(yW, sig(f32(ty) / p.overlap)); }
  if (p.tileY < p.tilesY - 1u) { yW = min(yW, sig(f32(p.curH - 1u - ty) / p.overlap)); }
  if (p.tileX > 0u)            { xW = min(xW, sig(f32(tx) / p.overlap)); }
  if (p.tileX < p.tilesX - 1u) { xW = min(xW, sig(f32(p.curW - 1u - tx) / p.overlap)); }
  let w = min(yW, xW);

  let stile = p.tileW * p.tileH;
  let sbase = p.batchIdx * 3u * stile;
  let sidx = ty * p.tileW + tx;
  let gplane = p.imgW * p.imgH;
  let gidx = (p.startY + ty) * p.imgW + (p.startX + tx);
  accum[0u * gplane + gidx] = accum[0u * gplane + gidx] + w * f32(src[sbase + 0u * stile + sidx]);
  accum[1u * gplane + gidx] = accum[1u * gplane + gidx] + w * f32(src[sbase + 1u * stile + sidx]);
  accum[2u * gplane + gidx] = accum[2u * gplane + gidx] + w * f32(src[sbase + 2u * stile + sidx]);
  weight[gidx] = weight[gidx] + w;
}
`;

const RESOLVE = /* wgsl */ `
struct P { imgW:u32, imgH:u32, srgb:u32, hdr:u32, flipY:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read> accum: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> p: P;

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let hi = pow(c, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
  let lo = c * 12.92;
  return select(lo, hi, c > vec3<f32>(0.0031308));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.imgW || gid.y >= p.imgH) { return; }
  let idx = gid.y * p.imgW + gid.x;
  let gplane = p.imgW * p.imgH;
  let w = weight[idx] + 1e-8;
  var rgb = vec3<f32>(accum[0u*gplane+idx], accum[1u*gplane+idx], accum[2u*gplane+idx]) / w;
  if (p.srgb == 1u) { rgb = linearToSrgb(rgb); }
  if (p.hdr == 0u) { rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)); }
  let oy = select(gid.y, p.imgH - 1u - gid.y, p.flipY == 1u);
  dst[oy * p.imgW + gid.x] = pack4x8unorm(vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

export interface AccumParams {
  imgW: number; imgH: number; startX: number; startY: number; curW: number; curH: number;
  tileX: number; tileY: number; tilesX: number; tilesY: number;
  tileW: number; tileH: number; overlap: number;
  batchIdx: number;
}

export class GpuImageOps {
  private extractPipe: GPUComputePipeline;
  private accumPipe: GPUComputePipeline;
  private resolvePipe: GPUComputePipeline;
  private extractParams: GPUBuffer; // 32B uniform (common)
  private extractOffsets: GPUBuffer; // maxBatch * 8B storage (per-slot startX/startY)
  private accumParams: GPUBuffer[]; // one 64B uniform per batch slot
  private resolveParams: GPUBuffer; // 32B

  constructor(private device: GPUDevice, readonly maxBatch: number, ioF16 = false) {
    const mk = (code: string) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
      });
    const io = ioF16 ? 'f16' : 'f32';
    this.extractPipe = mk(EXTRACT_TILES(io));
    this.accumPipe = mk(ACCUMULATE_TILE(io));
    this.resolvePipe = mk(RESOLVE);
    const u = (size: number) =>
      device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.extractParams = u(32);
    this.extractOffsets = device.createBuffer({
      size: Math.max(1, maxBatch) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.accumParams = Array.from({ length: Math.max(1, maxBatch) }, () => u(64));
    this.resolveParams = u(32);
  }

  private run(enc: GPUCommandEncoder, pipe: GPUComputePipeline, buffers: GPUBuffer[], dx: number, dy: number, dz = 1) {
    const bind = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(dx / 8), Math.ceil(dy / 8), dz);
    pass.end();
  }

  /** Extract `count` tiles (offsets = count pairs of startX,startY) into dst[B,C,tileH,tileW] in one dispatch. */
  encodeExtractTiles(
    enc: GPUCommandEncoder,
    color: GPUBuffer, albedo: GPUBuffer, normal: GPUBuffer, dst: GPUBuffer,
    imgW: number, imgH: number, tileW: number, tileH: number, channels: number, srgb: boolean,
    offsets: Uint32Array, count: number,
  ) {
    this.device.queue.writeBuffer(this.extractParams, 0,
      new Uint32Array([imgW, imgH, tileW, tileH, channels, srgb ? 1 : 0, count, 0]));
    this.device.queue.writeBuffer(this.extractOffsets, 0, offsets, 0, count * 2);
    this.run(enc, this.extractPipe,
      [color, albedo, normal, dst, this.extractParams, this.extractOffsets], tileW, tileH, count);
  }

  /**
   * Blend one batch-slot's output tile into accum/weight. Each call encodes its
   * own compute pass (overlapping tiles RMW the same texels; pass boundaries
   * order them). `slot` selects a dedicated uniform buffer so a whole batch of
   * accumulates can be encoded before a single submit.
   */
  encodeAccumulateTile(enc: GPUCommandEncoder, slot: number, outNCHW: GPUBuffer, accum: GPUBuffer, weight: GPUBuffer, p: AccumParams) {
    const ab = new ArrayBuffer(64);
    new Uint32Array(ab, 0, 13).set([
      p.imgW, p.imgH, p.startX, p.startY, p.curW, p.curH,
      p.tileX, p.tileY, p.tilesX, p.tilesY, p.tileW, p.tileH, p.batchIdx,
    ]);
    new Float32Array(ab, 52, 1)[0] = p.overlap;
    const params = this.accumParams[slot];
    this.device.queue.writeBuffer(params, 0, ab);
    this.run(enc, this.accumPipe, [outNCHW, accum, weight, params], p.curW, p.curH);
  }

  encodeResolve(
    enc: GPUCommandEncoder, accum: GPUBuffer, weight: GPUBuffer, dst: GPUBuffer,
    imgW: number, imgH: number, srgb: boolean, hdr: boolean, flipY = false,
  ) {
    this.device.queue.writeBuffer(this.resolveParams, 0,
      new Uint32Array([imgW, imgH, srgb ? 1 : 0, hdr ? 1 : 0, flipY ? 1 : 0, 0, 0, 0]));
    this.run(enc, this.resolvePipe, [accum, weight, dst, this.resolveParams], imgW, imgH);
  }
}
