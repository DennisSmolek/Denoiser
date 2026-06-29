// WGSL compute pre/post-processing on the shared GPUDevice — the GPU replacement
// for the old TensorFlow.js tensor math (normalization, layout, tiling, blend).
//
// Kernels:
//   extractTile  — pull a 256² tile (offset, zero-padded) from up to three full
//                  RGBA8 images (color [+albedo +normal]) into a planar NCHW input.
//                  color/albedo normalized to [0,1] (optionally sRGB->linear);
//                  normal mapped to OIDN's [-1,1].
//   accumulate   — blend a model-output tile into accum + weight buffers using the
//                  min-of-sigmoid overlap mask (matches the old tiler.ts).
//   resolve      — accum / weight -> RGBA8, optional linear->sRGB, LDR clamp.
//
// Layout is NCHW; channels is 3 (color), 6 (+albedo) or 9 (+albedo+normal).

const EXTRACT_TILE = /* wgsl */ `
struct P {
  imgW:u32, imgH:u32, startX:u32, startY:u32, tile:u32, channels:u32, srgb:u32, _pad:u32,
};
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<storage, read> albedo: array<u32>;
@group(0) @binding(2) var<storage, read> normal: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>; // NCHW, channels*tile*tile
@group(0) @binding(4) var<uniform> p: P;

fn srgbToLinear(c: vec3<f32>) -> vec3<f32> {
  let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
  let lo = c / 12.92;
  return select(lo, hi, c > vec3<f32>(0.04045));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.tile || gid.y >= p.tile) { return; }
  let plane = p.tile * p.tile;
  let didx = gid.y * p.tile + gid.x;
  let sx = p.startX + gid.x;
  let sy = p.startY + gid.y;
  let inside = sx < p.imgW && sy < p.imgH;
  let sidx = select(0u, sy * p.imgW + sx, inside);

  var col = vec3<f32>(0.0);
  if (inside) {
    col = unpack4x8unorm(color[sidx]).xyz;
    if (p.srgb == 1u) { col = srgbToLinear(col); }
  }
  dst[0u * plane + didx] = col.x;
  dst[1u * plane + didx] = col.y;
  dst[2u * plane + didx] = col.z;

  if (p.channels >= 6u) {
    var alb = vec3<f32>(0.0);
    if (inside) { alb = unpack4x8unorm(albedo[sidx]).xyz; }
    dst[3u * plane + didx] = alb.x;
    dst[4u * plane + didx] = alb.y;
    dst[5u * plane + didx] = alb.z;
  }
  if (p.channels >= 9u) {
    var nrm = vec3<f32>(0.0);
    if (inside) { nrm = unpack4x8unorm(normal[sidx]).xyz * 2.0 - 1.0; }
    dst[6u * plane + didx] = nrm.x;
    dst[7u * plane + didx] = nrm.y;
    dst[8u * plane + didx] = nrm.z;
  }
}
`;

const ACCUMULATE_TILE = /* wgsl */ `
struct P {
  imgW:u32, imgH:u32, startX:u32, startY:u32, curW:u32, curH:u32,
  tileX:u32, tileY:u32, tilesX:u32, tilesY:u32, tile:u32, _pad:u32, overlap:f32,
};
@group(0) @binding(0) var<storage, read> src: array<f32>;          // NCHW model output (3ch)
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

  let stile = p.tile * p.tile;
  let sidx = ty * p.tile + tx;
  let gplane = p.imgW * p.imgH;
  let gidx = (p.startY + ty) * p.imgW + (p.startX + tx);
  accum[0u * gplane + gidx] = accum[0u * gplane + gidx] + w * src[0u * stile + sidx];
  accum[1u * gplane + gidx] = accum[1u * gplane + gidx] + w * src[1u * stile + sidx];
  accum[2u * gplane + gidx] = accum[2u * gplane + gidx] + w * src[2u * stile + sidx];
  weight[gidx] = weight[gidx] + w;
}
`;

const RESOLVE = /* wgsl */ `
struct P { imgW:u32, imgH:u32, srgb:u32, hdr:u32 };
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
  dst[idx] = pack4x8unorm(vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

export interface AccumParams {
  imgW: number; imgH: number; startX: number; startY: number; curW: number; curH: number;
  tileX: number; tileY: number; tilesX: number; tilesY: number; tile: number; overlap: number;
}

export class GpuImageOps {
  private extractPipe: GPUComputePipeline;
  private accumPipe: GPUComputePipeline;
  private resolvePipe: GPUComputePipeline;
  private extractParams: GPUBuffer; // 32B
  private accumParams: GPUBuffer; // 64B
  private resolveParams: GPUBuffer; // 16B

  constructor(private device: GPUDevice) {
    const mk = (code: string) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
      });
    this.extractPipe = mk(EXTRACT_TILE);
    this.accumPipe = mk(ACCUMULATE_TILE);
    this.resolvePipe = mk(RESOLVE);
    const u = (size: number) =>
      device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.extractParams = u(32);
    this.accumParams = u(64);
    this.resolveParams = u(16);
  }

  private run(enc: GPUCommandEncoder, pipe: GPUComputePipeline, buffers: GPUBuffer[], dx: number, dy: number) {
    const bind = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(dx / 8), Math.ceil(dy / 8));
    pass.end();
  }

  encodeExtractTile(
    enc: GPUCommandEncoder,
    color: GPUBuffer, albedo: GPUBuffer, normal: GPUBuffer, dst: GPUBuffer,
    imgW: number, imgH: number, startX: number, startY: number, tile: number,
    channels: number, srgb: boolean,
  ) {
    this.device.queue.writeBuffer(this.extractParams, 0,
      new Uint32Array([imgW, imgH, startX, startY, tile, channels, srgb ? 1 : 0, 0]));
    this.run(enc, this.extractPipe, [color, albedo, normal, dst, this.extractParams], tile, tile);
  }

  encodeAccumulateTile(enc: GPUCommandEncoder, outNCHW: GPUBuffer, accum: GPUBuffer, weight: GPUBuffer, p: AccumParams) {
    const ab = new ArrayBuffer(64);
    new Uint32Array(ab, 0, 12).set([
      p.imgW, p.imgH, p.startX, p.startY, p.curW, p.curH,
      p.tileX, p.tileY, p.tilesX, p.tilesY, p.tile, 0,
    ]);
    new Float32Array(ab, 48, 1)[0] = p.overlap;
    this.device.queue.writeBuffer(this.accumParams, 0, ab);
    this.run(enc, this.accumPipe, [outNCHW, accum, weight, this.accumParams], p.curW, p.curH);
  }

  encodeResolve(
    enc: GPUCommandEncoder, accum: GPUBuffer, weight: GPUBuffer, dst: GPUBuffer,
    imgW: number, imgH: number, srgb: boolean, hdr: boolean,
  ) {
    this.device.queue.writeBuffer(this.resolveParams, 0,
      new Uint32Array([imgW, imgH, srgb ? 1 : 0, hdr ? 1 : 0]));
    this.run(enc, this.resolvePipe, [accum, weight, dst, this.resolveParams], imgW, imgH);
  }
}
