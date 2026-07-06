// WGSL compute pre/post-processing on the shared GPUDevice — the GPU replacement
// for the old TensorFlow.js tensor math (normalization, layout, tiling, blend).
//
// Kernels:
//   extractTiles — pull a BATCH of square tiles (per-tile offsets, zero-padded)
//                  from up to three full RGBA8 images (color [+albedo +normal])
//                  into a planar [B,C,tile,tile] NCHW input in one dispatch
//                  (workgroup z = batch slot). color/albedo normalized to [0,1]
//                  (optionally sRGB->linear); normals encoded to [0,1] à la
//                  upstream OIDN (docs/specs/oidn-color-reference.md).
//   accumulate   — blend ONE model-output tile (by batch slot) into accum +
//                  weight buffers using the min-of-sigmoid overlap mask
//                  (matches the old tiler.ts). Overlapping tiles must land in
//                  separate compute passes: pass boundaries synchronize the
//                  read-modify-write on accum/weight; z-batching them would race.
//   resolve      — accum / weight -> RGBA8, optional linear->sRGB, LDR clamp,
//                  optional Y flip.
//
// Layout is NCHW; channels is 3 (color), 6 (+albedo) or 9 (+albedo+normal).

// OIDN's PU transfer function for HDR color (docs/specs/oidn-color-reference.md):
// the network is trained on PU-encoded values — inputs go through
// pu_forward(y * inputScale) * PU_NORM, outputs through
// pu_inverse(x * PU_XMAX) / inputScale. inputScale comes from autoexposure
// (key 0.18 over the geometric mean luminance) via a 1-float storage buffer.
const PU_WGSL = /* wgsl */ `
const PU_A: f32 = 1.41283765e+03;
const PU_B: f32 = 1.64593172e+00;
const PU_C: f32 = 4.31384981e-01;
const PU_D: f32 = -2.94139609e-03;
const PU_E: f32 = 1.92653254e-01;
const PU_F: f32 = 6.26026094e-03;
const PU_G: f32 = 9.98620152e-01;
const PU_Y0: f32 = 1.57945760e-06;
const PU_Y1: f32 = 3.22087631e-02;
const PU_X0: f32 = 2.23151711e-03;
const PU_X1: f32 = 3.70974749e-01;
const PU_XMAX: f32 = 3.13512325;  // pu_forward1(65504) = PU_E*log(65504+PU_F)+PU_G
const PU_NORM: f32 = 0.318966;    // 1 / PU_XMAX

fn pu_forward1(y: f32) -> f32 {
  if (y <= PU_Y0) { return PU_A * y; }
  if (y <= PU_Y1) { return PU_B * pow(y, PU_C) + PU_D; }
  return PU_E * log(y + PU_F) + PU_G;
}
fn pu_inverse1(x: f32) -> f32 {
  if (x <= PU_X0) { return x / PU_A; }
  if (x <= PU_X1) { return pow((x - PU_D) / PU_B, 1.0 / PU_C); }
  return exp((x - PU_G) / PU_E) - PU_F;
}
fn pu_forward(y: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(pu_forward1(y.x), pu_forward1(y.y), pu_forward1(y.z)) * PU_NORM;
}
fn pu_inverse(x: vec3<f32>) -> vec3<f32> {
  let xs = x * PU_XMAX;
  return vec3<f32>(pu_inverse1(xs.x), pu_inverse1(xs.y), pu_inverse1(xs.z));
}
`;

// `io` is the model IO element type: 'f32', or 'f16' for fp16 models (needs the
// shader-f16 device feature). Only the model-facing NCHW buffers change type;
// accum/weight/resolve stay f32.
const EXTRACT_TILES = (io: string) => /* wgsl */ `
${io === 'f16' ? 'enable f16;' : ''}
alias IOType = ${io};
${PU_WGSL}
struct P {
  imgW:u32, imgH:u32, tileW:u32, tileH:u32, channels:u32, srgb:u32, count:u32, hdr:u32,
};
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<storage, read> albedo: array<u32>;
@group(0) @binding(2) var<storage, read> normal: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<IOType>; // NCHW, count*channels*tileW*tileH
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> offsets: array<vec2<u32>>; // per-slot startX,startY
@group(0) @binding(6) var<storage, read> exposure: array<f32>; // [inputScale] (autoexposure)

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
    if (p.hdr == 1u) { col = pu_forward(max(col * exposure[0], vec3<f32>(0.0))); }
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
    // OIDN feeds the network normals ENCODED to [0,1] (clamp(n,-1,1)*0.5+0.5 —
    // see docs/specs/oidn-color-reference.md). RGBA8 bytes already hold that encoding;
    // pad with 0.5 (the encoded zero-normal).
    var nrm = vec3<f32>(0.5);
    if (inside) { nrm = unpack4x8unorm(normal[sidx]).xyz; }
    dst[base + 6u * plane + didx] = IOType(nrm.x);
    dst[base + 7u * plane + didx] = IOType(nrm.y);
    dst[base + 8u * plane + didx] = IOType(nrm.z);
  }
}
`;

// Texture-input variant: reads float textures (e.g. a path tracer's linear-HDR
// render target) instead of RGBA8 storage buffers — no CPU round-trip, no 8-bit
// quantization. Color passes through as-is (hdr) or sRGB->linear (srgb flag);
// albedo expected [0,1]; normal expected already [-1,1] (G-buffer convention).
// flipY reads the source bottom-up (WebGPU render targets).
const EXTRACT_TILES_TEX = (io: string) => /* wgsl */ `
${io === 'f16' ? 'enable f16;' : ''}
alias IOType = ${io};
${PU_WGSL}
struct P {
  imgW:u32, imgH:u32, tileW:u32, tileH:u32, channels:u32, srgb:u32, count:u32, flipY:u32,
  hdr:u32, auxFlipY:u32, _p1:u32, _p2:u32,
};
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var normal: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<IOType>; // NCHW, count*channels*tileW*tileH
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> offsets: array<vec2<u32>>; // per-slot startX,startY
@group(0) @binding(6) var<storage, read> exposure: array<f32>; // [inputScale] (autoexposure)

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
  let ly = select(sy, p.imgH - 1u - sy, p.flipY == 1u);
  let coord = vec2<i32>(i32(sx), i32(ly));
  // aux sources can have the opposite vertical convention (e.g. raster G-buffer
  // vs compute-written tracer output) — separate flip
  let lyAux = select(sy, p.imgH - 1u - sy, p.auxFlipY == 1u);
  let coordAux = vec2<i32>(i32(sx), i32(lyAux));

  var col = vec3<f32>(0.0);
  if (inside) {
    // float texture inputs are linear by contract; p.srgb is output-side only
    col = textureLoad(color, coord, 0).xyz;
    if (p.hdr == 1u) { col = pu_forward(max(col * exposure[0], vec3<f32>(0.0))); }
  }
  dst[base + 0u * plane + didx] = IOType(col.x);
  dst[base + 1u * plane + didx] = IOType(col.y);
  dst[base + 2u * plane + didx] = IOType(col.z);

  if (p.channels >= 6u) {
    var alb = vec3<f32>(0.0);
    if (inside) { alb = textureLoad(albedo, coordAux, 0).xyz; }
    dst[base + 3u * plane + didx] = IOType(alb.x);
    dst[base + 4u * plane + didx] = IOType(alb.y);
    dst[base + 5u * plane + didx] = IOType(alb.z);
  }
  if (p.channels >= 9u) {
    // float G-buffer normals arrive [-1,1]; the network wants them encoded [0,1]
    var nrm = vec3<f32>(0.5);
    if (inside) {
      nrm = clamp(textureLoad(normal, coordAux, 0).xyz, vec3<f32>(-1.0), vec3<f32>(1.0)) * 0.5 + 0.5;
    }
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

// Shared resolve math: accum/weight -> display rgb. tonemap = Narkowicz ACES
// (for HDR results headed straight to a canvas) applied before the sRGB encode.
// Autoexposure (upstream OIDN algorithm): per-16px-bin mean luminance, then
// inputScale = 0.18 / geometric-mean of the bin luminances.
const AUTOEXPOSURE_BINS = /* wgsl */ `
struct P { imgW:u32, imgH:u32, binsX:u32, binsY:u32 };
@group(0) @binding(0) var color: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let x0 = (wid.x * p.imgW) / p.binsX; let x1 = ((wid.x + 1u) * p.imgW) / p.binsX;
  let y0 = (wid.y * p.imgH) / p.binsY; let y1 = ((wid.y + 1u) * p.imgH) / p.binsY;
  var sum = 0.0;
  var yy = y0 + lid.y;
  while (yy < y1) {
    var xx = x0 + lid.x;
    while (xx < x1) {
      let c = clamp(textureLoad(color, vec2<i32>(i32(xx), i32(yy)), 0).xyz,
                    vec3<f32>(0.0), vec3<f32>(3.4e38));
      sum += 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
      xx += 8u;
    }
    yy += 8u;
  }
  partial[li] = sum;
  workgroupBarrier();
  var s = 32u;
  while (s > 0u) {
    if (li < s) { partial[li] += partial[li + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (li == 0u) {
    let count = f32(max((x1 - x0) * (y1 - y0), 1u));
    bins[wid.y * p.binsX + wid.x] = partial[0] / count;
  }
}
`;

const AUTOEXPOSURE_REDUCE = /* wgsl */ `
struct P { numBins:u32, _0:u32, _1:u32, _2:u32 };
@group(0) @binding(0) var<storage, read> bins: array<f32>;
@group(0) @binding(1) var<storage, read_write> exposure: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  var count = 0.0;
  for (var i = 0u; i < p.numBins; i++) {
    if (bins[i] > 1e-8) { sum += log2(bins[i]); count += 1.0; }
  }
  exposure[0] = select(1.0, 0.18 / exp2(sum / count), count > 0.0);
}
`;

const RESOLVE_COMMON = /* wgsl */ `
${PU_WGSL}
struct P { imgW:u32, imgH:u32, srgb:u32, hdr:u32, flipY:u32, tonemap:u32, _p1:u32, _p2:u32 };

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let hi = pow(c, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
  let lo = c * 12.92;
  return select(lo, hi, c > vec3<f32>(0.0031308));
}

fn acesTonemap(c: vec3<f32>) -> vec3<f32> {
  return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn resolveRgb(rgbIn: vec3<f32>, p: P) -> vec3<f32> {
  var rgb = rgbIn;
  if (p.tonemap == 1u) { rgb = linearToSrgb(acesTonemap(rgb)); }
  else {
    if (p.srgb == 1u) { rgb = linearToSrgb(rgb); }
    if (p.hdr == 0u) { rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)); }
  }
  return rgb; // NOTE: unclamped — hdr float outputs keep their range; unorm sinks clamp
}
`;

const RESOLVE = /* wgsl */ `
${RESOLVE_COMMON}
@group(0) @binding(0) var<storage, read> accum: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> exposure: array<f32>; // [inputScale]

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.imgW || gid.y >= p.imgH) { return; }
  let idx = gid.y * p.imgW + gid.x;
  let gplane = p.imgW * p.imgH;
  let w = weight[idx] + 1e-8;
  var rgb = vec3<f32>(accum[0u*gplane+idx], accum[1u*gplane+idx], accum[2u*gplane+idx]) / w;
  if (p.hdr == 1u) { rgb = pu_inverse(max(rgb, vec3<f32>(0.0))) / exposure[0]; }
  rgb = clamp(resolveRgb(rgb, p), vec3<f32>(0.0), vec3<f32>(1.0));
  let oy = select(gid.y, p.imgH - 1u - gid.y, p.flipY == 1u);
  dst[oy * p.imgW + gid.x] = pack4x8unorm(vec4<f32>(rgb, 1.0));
}
`;

// format: rgba8unorm (clamped, display-ready) or rgba16float (unclamped — keeps
// HDR range so e.g. three.js can tonemap in its own pipeline).
const RESOLVE_TEX = (format: string) => /* wgsl */ `
${RESOLVE_COMMON}
@group(0) @binding(0) var<storage, read> accum: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<${format}, write>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> exposure: array<f32>; // [inputScale]

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.imgW || gid.y >= p.imgH) { return; }
  let idx = gid.y * p.imgW + gid.x;
  let gplane = p.imgW * p.imgH;
  let w = weight[idx] + 1e-8;
  var rgb = vec3<f32>(accum[0u*gplane+idx], accum[1u*gplane+idx], accum[2u*gplane+idx]) / w;
  if (p.hdr == 1u) { rgb = pu_inverse(max(rgb, vec3<f32>(0.0))) / exposure[0]; }
  rgb = resolveRgb(rgb, p);
  ${format === 'rgba8unorm' ? 'rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));' : ''}
  let oy = select(gid.y, p.imgH - 1u - gid.y, p.flipY == 1u);
  textureStore(dst, vec2<i32>(i32(gid.x), i32(oy)), vec4<f32>(rgb, 1.0));
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
  private extractTexPipe?: GPUComputePipeline; // lazy — texture-input path
  private resolveTexPipes = new Map<string, GPUComputePipeline>(); // lazy, per format
  private extractParams: GPUBuffer; // 48B uniform (common; TEX variant uses 12 u32)
  private extractOffsets: GPUBuffer; // maxBatch * 8B storage (per-slot startX/startY)
  private accumParams: GPUBuffer[]; // one 64B uniform per batch slot
  private resolveParams: GPUBuffer; // 32B
  private exposureBuf: GPUBuffer; // [inputScale] — autoexposure result or manual value
  private binsBuf?: GPUBuffer; // autoexposure bin luminances
  private binsCapacity = 0;
  private aeBinsPipe?: GPUComputePipeline;
  private aeReducePipe?: GPUComputePipeline;
  private aeBinsParams?: GPUBuffer;
  private aeReduceParams?: GPUBuffer;
  private readonly io: string;

  private mk(code: string) {
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' },
    });
  }

  constructor(private device: GPUDevice, readonly maxBatch: number, ioF16 = false) {
    const mk = (code: string) => this.mk(code);
    const io = (this.io = ioF16 ? 'f16' : 'f32');
    this.extractPipe = mk(EXTRACT_TILES(io));
    this.accumPipe = mk(ACCUMULATE_TILE(io));
    this.resolvePipe = mk(RESOLVE);
    const u = (size: number) =>
      device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.exposureBuf = device.createBuffer({
      size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.setExposure(1);
    this.extractParams = u(48);
    this.extractOffsets = device.createBuffer({
      size: Math.max(1, maxBatch) * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.accumParams = Array.from({ length: Math.max(1, maxBatch) }, () => u(64));
    this.resolveParams = u(32);
  }

  private run(enc: GPUCommandEncoder, pipe: GPUComputePipeline, buffers: GPUBuffer[], dx: number, dy: number, dz = 1) {
    this.runMixed(enc, pipe, buffers.map((buffer) => ({ buffer })), dx, dy, dz);
  }

  private runMixed(enc: GPUCommandEncoder, pipe: GPUComputePipeline, resources: GPUBindingResource[], dx: number, dy: number, dz = 1) {
    const bind = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: resources.map((resource, i) => ({ binding: i, resource })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(dx / 8), Math.ceil(dy / 8), dz);
    pass.end();
  }

  /** Set the HDR input scale manually (autoexposure overwrites it when encoded). */
  setExposure(inputScale: number) {
    this.device.queue.writeBuffer(this.exposureBuf, 0, new Float32Array([inputScale]));
  }

  /** OIDN autoexposure: computes inputScale from the color texture into the exposure buffer. */
  encodeAutoexposure(enc: GPUCommandEncoder, color: GPUTextureView, imgW: number, imgH: number) {
    this.aeBinsPipe ??= this.mk(AUTOEXPOSURE_BINS);
    this.aeReducePipe ??= this.mk(AUTOEXPOSURE_REDUCE);
    const u = (size: number) =>
      this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.aeBinsParams ??= u(16);
    this.aeReduceParams ??= u(16);
    const binsX = Math.ceil(imgW / 16);
    const binsY = Math.ceil(imgH / 16);
    if (binsX * binsY > this.binsCapacity) {
      this.binsBuf?.destroy();
      this.binsCapacity = binsX * binsY;
      this.binsBuf = this.device.createBuffer({
        size: this.binsCapacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.aeBinsParams, 0, new Uint32Array([imgW, imgH, binsX, binsY]));
    this.device.queue.writeBuffer(this.aeReduceParams, 0, new Uint32Array([binsX * binsY, 0, 0, 0]));
    this.runMixed(enc, this.aeBinsPipe,
      [color, { buffer: this.binsBuf! }, { buffer: this.aeBinsParams }], binsX * 8, binsY * 8);
    this.runMixed(enc, this.aeReducePipe,
      [{ buffer: this.binsBuf! }, { buffer: this.exposureBuf }, { buffer: this.aeReduceParams }], 1, 1);
  }

  /** Extract `count` tiles (offsets = count pairs of startX,startY) into dst[B,C,tileH,tileW] in one dispatch. */
  encodeExtractTiles(
    enc: GPUCommandEncoder,
    color: GPUBuffer, albedo: GPUBuffer, normal: GPUBuffer, dst: GPUBuffer,
    imgW: number, imgH: number, tileW: number, tileH: number, channels: number, srgb: boolean,
    hdr: boolean, offsets: Uint32Array, count: number,
  ) {
    this.device.queue.writeBuffer(this.extractParams, 0,
      new Uint32Array([imgW, imgH, tileW, tileH, channels, srgb ? 1 : 0, count, hdr ? 1 : 0]));
    this.device.queue.writeBuffer(this.extractOffsets, 0, offsets, 0, count * 2);
    this.run(enc, this.extractPipe,
      [color, albedo, normal, dst, this.extractParams, this.extractOffsets, this.exposureBuf],
      tileW, tileH, count);
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
    imgW: number, imgH: number, srgb: boolean, hdr: boolean, flipY = false, tonemap = false,
  ) {
    this.device.queue.writeBuffer(this.resolveParams, 0,
      new Uint32Array([imgW, imgH, srgb ? 1 : 0, hdr ? 1 : 0, flipY ? 1 : 0, tonemap ? 1 : 0, 0, 0]));
    this.run(enc, this.resolvePipe, [accum, weight, dst, this.resolveParams, this.exposureBuf], imgW, imgH);
  }

  /** Texture-input extract: color/albedo/normal are float texture views. */
  encodeExtractTilesTex(
    enc: GPUCommandEncoder,
    color: GPUTextureView, albedo: GPUTextureView, normal: GPUTextureView, dst: GPUBuffer,
    imgW: number, imgH: number, tileW: number, tileH: number, channels: number,
    srgb: boolean, flipY: boolean, hdr: boolean, auxFlipY: boolean,
    offsets: Uint32Array, count: number,
  ) {
    this.extractTexPipe ??= this.mk(EXTRACT_TILES_TEX(this.io));
    this.device.queue.writeBuffer(this.extractParams, 0,
      new Uint32Array([imgW, imgH, tileW, tileH, channels, srgb ? 1 : 0, count, flipY ? 1 : 0,
        hdr ? 1 : 0, auxFlipY ? 1 : 0, 0, 0]));
    this.device.queue.writeBuffer(this.extractOffsets, 0, offsets, 0, count * 2);
    this.runMixed(enc, this.extractTexPipe,
      [color, albedo, normal, { buffer: dst }, { buffer: this.extractParams },
        { buffer: this.extractOffsets }, { buffer: this.exposureBuf }],
      tileW, tileH, count);
  }

  /** Resolve straight into a storage texture (no CPU readback). rgba8unorm or rgba16float. */
  encodeResolveToTexture(
    enc: GPUCommandEncoder, accum: GPUBuffer, weight: GPUBuffer, dst: GPUTextureView,
    format: string, imgW: number, imgH: number,
    srgb: boolean, hdr: boolean, flipY = false, tonemap = false,
  ) {
    let pipe = this.resolveTexPipes.get(format);
    if (!pipe) {
      pipe = this.mk(RESOLVE_TEX(format));
      this.resolveTexPipes.set(format, pipe);
    }
    this.device.queue.writeBuffer(this.resolveParams, 0,
      new Uint32Array([imgW, imgH, srgb ? 1 : 0, hdr ? 1 : 0, flipY ? 1 : 0, tonemap ? 1 : 0, 0, 0]));
    this.runMixed(enc, pipe,
      [{ buffer: accum }, { buffer: weight }, dst, { buffer: this.resolveParams },
        { buffer: this.exposureBuf }], imgW, imgH);
  }
}
