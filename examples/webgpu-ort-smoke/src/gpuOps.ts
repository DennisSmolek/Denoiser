// WGSL compute pre/post-processing on the shared GPUDevice.
//
// Replaces the TensorFlow.js tensor math the old library used for normalization
// and HWC<->NCHW layout. Two kernels:
//   - rgbaToNCHW: RGBA8 pixels (one u32/pixel) -> planar NCHW f32. unpack4x8unorm
//     gives the /255 normalization for free. Optional normal-map mode maps [0,1]->[-1,1].
//   - nchwToRGBA: planar NCHW f32 (3ch) -> RGBA8 pixels, clamped to [0,1].
//
// Everything stays on the GPU; the only readback is the final pixels for a 2D
// canvas (in the real three.js pipeline that becomes a texture write instead).

const RGBA_TO_NCHW = /* wgsl */ `
struct Params { w: u32, h: u32, channels: u32, normalMode: u32 };
@group(0) @binding(0) var<storage, read> src: array<u32>;       // RGBA8, w*h
@group(0) @binding(1) var<storage, read_write> dst: array<f32>; // NCHW, channels*w*h
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let idx = gid.y * p.w + gid.x;
  var rgba = unpack4x8unorm(src[idx]);          // [0,1]
  if (p.normalMode == 1u) { rgba = rgba * 2.0 - 1.0; } // OIDN normal range [-1,1]
  let plane = p.w * p.h;
  dst[0u * plane + idx] = rgba.x;
  dst[1u * plane + idx] = rgba.y;
  dst[2u * plane + idx] = rgba.z;
}
`;

const NCHW_TO_RGBA = /* wgsl */ `
struct Params { w: u32, h: u32, channels: u32, normalMode: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;       // NCHW, 3*w*h
@group(0) @binding(1) var<storage, read_write> dst: array<u32>; // RGBA8, w*h
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let idx = gid.y * p.w + gid.x;
  let plane = p.w * p.h;
  let rgb = vec3<f32>(src[0u * plane + idx], src[1u * plane + idx], src[2u * plane + idx]);
  dst[idx] = pack4x8unorm(vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

// Tiling kernels (mirror packages/denoiser/src/tiler.ts): 256² tiles, overlap 32
// (stride 224), edge tiles zero-padded, blended with min-of-sigmoid ramps.
const EXTRACT_TILE = /* wgsl */ `
struct P { imgW:u32, imgH:u32, startX:u32, startY:u32, tile:u32, normalMode:u32 };
@group(0) @binding(0) var<storage, read> src: array<u32>;       // full RGBA8 image
@group(0) @binding(1) var<storage, read_write> dst: array<f32>; // NCHW tile (3*tile*tile)
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.tile || gid.y >= p.tile) { return; }
  let plane = p.tile * p.tile;
  let didx = gid.y * p.tile + gid.x;
  let sx = p.startX + gid.x;
  let sy = p.startY + gid.y;
  var rgb = vec3<f32>(0.0);              // zero-pad outside the image (matches tf.pad)
  if (sx < p.imgW && sy < p.imgH) {
    var rgba = unpack4x8unorm(src[sy * p.imgW + sx]);
    if (p.normalMode == 1u) { rgba = rgba * 2.0 - 1.0; }
    rgb = rgba.xyz;
  }
  dst[0u * plane + didx] = rgb.x;
  dst[1u * plane + didx] = rgb.y;
  dst[2u * plane + didx] = rgb.z;
}
`;

const ACCUMULATE_TILE = /* wgsl */ `
struct P {
  imgW:u32, imgH:u32, startX:u32, startY:u32, curW:u32, curH:u32,
  tileX:u32, tileY:u32, tilesX:u32, tilesY:u32, tile:u32, _pad:u32, overlap:f32,
};
@group(0) @binding(0) var<storage, read> src: array<f32>;          // NCHW model output
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
struct P { imgW:u32, imgH:u32 };
@group(0) @binding(0) var<storage, read> accum: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.imgW || gid.y >= p.imgH) { return; }
  let idx = gid.y * p.imgW + gid.x;
  let gplane = p.imgW * p.imgH;
  let w = weight[idx] + 1e-8;
  let rgb = vec3<f32>(accum[0u*gplane+idx], accum[1u*gplane+idx], accum[2u*gplane+idx]) / w;
  dst[idx] = pack4x8unorm(vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

// Concatenate color + albedo + normal into a 9-channel NCHW input (single tile).
// color/albedo normalized to [0,1]; normal mapped to OIDN's [-1,1].
const RGBA_TO_NCHW_AUX = /* wgsl */ `
struct P { w: u32, h: u32 };
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<storage, read> albedo: array<u32>;
@group(0) @binding(2) var<storage, read> normal: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>; // 9*w*h
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let idx = gid.y * p.w + gid.x;
  let plane = p.w * p.h;
  let c = unpack4x8unorm(color[idx]);
  let a = unpack4x8unorm(albedo[idx]);
  let n = unpack4x8unorm(normal[idx]) * 2.0 - 1.0;
  dst[0u * plane + idx] = c.x; dst[1u * plane + idx] = c.y; dst[2u * plane + idx] = c.z;
  dst[3u * plane + idx] = a.x; dst[4u * plane + idx] = a.y; dst[5u * plane + idx] = a.z;
  dst[6u * plane + idx] = n.x; dst[7u * plane + idx] = n.y; dst[8u * plane + idx] = n.z;
}
`;

export class GpuImageOps {
  private toNCHWPipe: GPUComputePipeline;
  private auxPipe?: GPUComputePipeline;
  private toRGBAPipe: GPUComputePipeline;
  private extractPipe: GPUComputePipeline;
  private accumPipe: GPUComputePipeline;
  private resolvePipe: GPUComputePipeline;
  private params: GPUBuffer;
  private tileParams: GPUBuffer; // 16-byte uniform for extract/resolve
  private accumParams: GPUBuffer; // 64-byte uniform for accumulate

  constructor(private device: GPUDevice) {
    this.toNCHWPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: RGBA_TO_NCHW }), entryPoint: 'main' },
    });
    this.toRGBAPipe = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: NCHW_TO_RGBA }), entryPoint: 'main' },
    });
    const mk = (code: string) =>
      device.createComputePipeline({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
      });
    this.extractPipe = mk(EXTRACT_TILE);
    this.accumPipe = mk(ACCUMULATE_TILE);
    this.resolvePipe = mk(RESOLVE);
    this.params = device.createBuffer({
      size: 16, // vec4<u32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.tileParams = device.createBuffer({
      size: 32, // P{imgW,imgH,startX,startY,tile,normalMode} padded
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.accumParams = device.createBuffer({
      size: 64, // 12 u32 + 1 f32, padded to 16
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private setParams(w: number, h: number, channels: number, normalMode: number) {
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([w, h, channels, normalMode]));
  }

  private dispatch(
    enc: GPUCommandEncoder,
    pipe: GPUComputePipeline,
    src: GPUBuffer,
    dst: GPUBuffer,
    w: number,
    h: number,
  ) {
    const bind = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: this.params } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
  }

  /** RGBA8 buffer -> NCHW f32 buffer (normalized). normalMode=1 for normal maps. */
  encodeToNCHW(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, w: number, h: number, normalMode = 0) {
    this.setParams(w, h, 3, normalMode);
    this.dispatch(enc, this.toNCHWPipe, src, dst, w, h);
  }

  /** NCHW f32 buffer (3ch) -> RGBA8 buffer (clamped). */
  encodeToRGBA(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer, w: number, h: number) {
    this.setParams(w, h, 3, 0);
    this.dispatch(enc, this.toRGBAPipe, src, dst, w, h);
  }

  /** color+albedo+normal RGBA8 buffers -> 9-channel NCHW input (single tile). */
  encodeToNCHWAux(
    enc: GPUCommandEncoder, color: GPUBuffer, albedo: GPUBuffer, normal: GPUBuffer,
    dst: GPUBuffer, w: number, h: number,
  ) {
    if (!this.auxPipe) {
      this.auxPipe = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: this.device.createShaderModule({ code: RGBA_TO_NCHW_AUX }), entryPoint: 'main' },
      });
    }
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([w, h, 0, 0]));
    this.bindAndRun(enc, this.auxPipe, [color, albedo, normal, dst, this.params], w, h);
  }

  private bindAndRun(
    enc: GPUCommandEncoder,
    pipe: GPUComputePipeline,
    buffers: GPUBuffer[],
    dispatchW: number,
    dispatchH: number,
  ) {
    const bind = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(dispatchW / 8), Math.ceil(dispatchH / 8));
    pass.end();
  }

  /** Extract a tile from the full image into the NCHW model-input buffer (zero-padded, normalized). */
  encodeExtractTile(
    enc: GPUCommandEncoder, img: GPUBuffer, nchw: GPUBuffer,
    imgW: number, imgH: number, startX: number, startY: number, tile: number, normalMode = 0,
  ) {
    this.device.queue.writeBuffer(
      this.tileParams, 0, new Uint32Array([imgW, imgH, startX, startY, tile, normalMode]));
    this.bindAndRun(enc, this.extractPipe, [img, nchw, this.tileParams], tile, tile);
  }

  /** Blend a model-output tile into the accumulation + weight buffers (min-of-sigmoid mask). */
  encodeAccumulateTile(
    enc: GPUCommandEncoder, outNCHW: GPUBuffer, accum: GPUBuffer, weight: GPUBuffer,
    p: { imgW: number; imgH: number; startX: number; startY: number; curW: number; curH: number;
         tileX: number; tileY: number; tilesX: number; tilesY: number; tile: number; overlap: number },
  ) {
    const ab = new ArrayBuffer(64);
    new Uint32Array(ab, 0, 12).set([
      p.imgW, p.imgH, p.startX, p.startY, p.curW, p.curH,
      p.tileX, p.tileY, p.tilesX, p.tilesY, p.tile, 0,
    ]);
    new Float32Array(ab, 48, 1)[0] = p.overlap;
    this.device.queue.writeBuffer(this.accumParams, 0, ab);
    this.bindAndRun(enc, this.accumPipe, [outNCHW, accum, weight, this.accumParams], p.curW, p.curH);
  }

  /** accum / weight -> RGBA8 output. */
  encodeResolve(
    enc: GPUCommandEncoder, accum: GPUBuffer, weight: GPUBuffer, dst: GPUBuffer,
    imgW: number, imgH: number,
  ) {
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([imgW, imgH, 0, 0]));
    this.bindAndRun(enc, this.resolvePipe, [accum, weight, dst, this.params], imgW, imgH);
  }
}
