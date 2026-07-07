// The kernel registry. Every variant: { name, note, code(p), dispatch(p) }
// plus optional flags the harness honors:
//   f16:   bind f16 copies of input/weights/bias and an f16 output buffer
//   relu6: correctness-compare against clamp(ref, 0, 6) instead of ref
//   tol:   max-abs-diff gate (default 1e-3)
// Contract (f32): bindings 0=input (NCHW, [1,CIN,H,W]), 1=weights (OIHW),
// 2=bias ([COUT]), 3=output (NCHW, [1,COUT,H,W]), 4=uniform {W,H,CIN,COUT}.
// Add ideas freely — the harness correctness-gates and times each one; a bad
// kernel is a data point, not a problem.

// --- codegen: tiled shared-memory 3x3 conv ---------------------------------
// 16x16 workgroup = one 16x16 output tile. Input tile + 1px halo (18x18) is
// staged in workgroup memory in chunks of `cch` input channels; the weights
// for this workgroup's `cob` output channels x chunk are staged alongside.
// Each thread owns one output pixel and `cob` accumulators (unrolled as
// scalar vars — dynamic-indexed arrays spill to private memory). dispatch.z
// covers COUT/cob output-channel blocks.
// Workgroup storage: 18*18*cch*bpe + cob*cch*9*bpe, kept under the default
// 16KB maxComputeWorkgroupStorageSize (ORT owns device creation).
// opts.acc32 (f16 only): 9-term dot in f16, accumulate in f32.
function tiledCode(opts) {
  const { cob, cch, f16 = false, acc32 = false, relu6 = false } = opts;
  const T = f16 ? 'f16' : 'f32';
  const A = f16 && !acc32 ? 'f16' : 'f32';
  const smem = (324 * cch + cob * cch * 9) * (f16 ? 2 : 4);
  if (smem > 16384) throw new Error(`workgroup storage ${smem} > 16384`);
  const n = [...Array(cob).keys()];
  const accDecl = n.map((i) => `  var a${i}: ${A} = ${A}(bias[coBase + ${i}u]);`).join('\n');
  const macs = n.map((i) => {
    const w = (k) => `wsm[${i * cch * 9}u + wb + ${k}u]`;
    const dot = `i00*${w(0)} + i01*${w(1)} + i02*${w(2)} + i10*${w(3)} + i11*${w(4)} + i12*${w(5)} + i20*${w(6)} + i21*${w(7)} + i22*${w(8)}`;
    return `      a${i} += ${acc32 && f16 ? `f32(${dot})` : dot};`;
  }).join('\n');
  const out = n.map((i) => {
    const v = relu6 ? `clamp(a${i}, ${A}(0.0), ${A}(6.0))` : `a${i}`;
    return `    output[(coBase + ${i}u) * plane + oy * p.w + ox] = ${T}(${v});`;
  }).join('\n');
  return /* wgsl */ `
${f16 ? 'enable f16;' : ''}
struct P { w:u32, h:u32, cin:u32, cout:u32 };
@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read> weights: array<${T}>;
@group(0) @binding(2) var<storage, read> bias: array<${T}>;
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> tile: array<${T}, ${324 * cch}>;   // 18x18 x cch
var<workgroup> wsm: array<${T}, ${cob * cch * 9}>; // cob x cch x 3x3

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  let tx0 = wg.x * 16u;
  let ty0 = wg.y * 16u;
  let coBase = wg.z * ${cob}u;
  let plane = p.w * p.h;
${accDecl}
  for (var c0 = 0u; c0 < p.cin; c0 += ${cch}u) {
    // stage 18x18 x cch input tile (zero-padded halo)
    for (var i = lidx; i < ${324 * cch}u; i += 256u) {
      let ci = i / 324u;
      let r = i % 324u;
      let gy = i32(ty0 + r / 18u) - 1;
      let gx = i32(tx0 + r % 18u) - 1;
      var v = ${T}(0.0);
      if (gx >= 0 && gx < i32(p.w) && gy >= 0 && gy < i32(p.h)) {
        v = input[(c0 + ci) * plane + u32(gy) * p.w + u32(gx)];
      }
      tile[i] = v;
    }
    // stage cob x cch x 9 weights
    for (var i = lidx; i < ${cob * cch * 9}u; i += 256u) {
      let co = i / ${cch * 9}u;
      let r = i % ${cch * 9}u;
      wsm[i] = weights[((coBase + co) * p.cin + c0 + r / 9u) * 9u + r % 9u];
    }
    workgroupBarrier();
    for (var ci = 0u; ci < ${cch}u; ci++) {
      let tb = ci * 324u + li.y * 18u + li.x; // top-left of this pixel's 3x3
      let i00 = tile[tb];       let i01 = tile[tb + 1u];  let i02 = tile[tb + 2u];
      let i10 = tile[tb + 18u]; let i11 = tile[tb + 19u]; let i12 = tile[tb + 20u];
      let i20 = tile[tb + 36u]; let i21 = tile[tb + 37u]; let i22 = tile[tb + 38u];
      let wb = ci * 9u;
${macs}
    }
    workgroupBarrier();
  }
  let ox = tx0 + li.x;
  let oy = ty0 + li.y;
  if (ox < p.w && oy < p.h) {
${out}
  }
}`;
}

function tiled(name, note, opts) {
  return {
    name,
    note,
    f16: !!opts.f16,
    relu6: !!opts.relu6,
    tol: opts.f16 ? 1e-1 : 1e-3,
    code: () => tiledCode(opts),
    dispatch: (p) => [Math.ceil(p.w / 16), Math.ceil(p.h / 16), Math.ceil(p.cout / opts.cob)],
  };
}

export const kernels = [
  {
    name: 'naive',
    note: 'one thread per (x,y,cout), scalar loops — correctness anchor',
    code: () => /* wgsl */ `
struct P { w:u32, h:u32, cin:u32, cout:u32 };
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: P;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= p.w || g.y >= p.h || g.z >= p.cout) { return; }
  let plane = p.w * p.h;
  var acc = bias[g.z];
  for (var ci = 0u; ci < p.cin; ci++) {
    let wbase = ((g.z * p.cin) + ci) * 9u;
    for (var ky = 0u; ky < 3u; ky++) {
      let sy = i32(g.y) + i32(ky) - 1;
      if (sy < 0 || sy >= i32(p.h)) { continue; }
      for (var kx = 0u; kx < 3u; kx++) {
        let sx = i32(g.x) + i32(kx) - 1;
        if (sx < 0 || sx >= i32(p.w)) { continue; }
        acc += input[ci * plane + u32(sy) * p.w + u32(sx)] * weights[wbase + ky * 3u + kx];
      }
    }
  }
  output[g.z * plane + g.y * p.w + g.x] = acc;
}`,
    dispatch: (p) => [Math.ceil(p.w / 8), Math.ceil(p.h / 8), p.cout],
  },
  tiled('tiled-smem', '16x16 tile+halo smem, 8ch chunks, 8 cout/thread', { cob: 8, cch: 8 }),
  tiled('tiled-smem-co16', 'same, 16 cout/thread (fewer tile reloads)', { cob: 16, cch: 8 }),
  tiled('tiled-smem-co32', '32 cout/thread, 4ch chunks (smem cap)', { cob: 32, cch: 4 }),
  tiled('tiled-f16', 'f16 storage+math, 16ch chunks, 16 cout/thread', { cob: 16, cch: 16, f16: true }),
  tiled('tiled-f16-co32', 'f16, 32 cout/thread, 8ch chunks', { cob: 32, cch: 8, f16: true }),
  tiled('tiled-f16-acc32', 'f16 loads/dots, f32 accumulate (accuracy check)', { cob: 16, cch: 16, f16: true, acc32: true }),
  tiled('fused-relu6', 'relu6 folded into best f32 tiled epilogue', { cob: 16, cch: 8, relu6: true }),
  tiled('fused-relu6-f16', 'relu6 folded into best f16 tiled epilogue', { cob: 32, cch: 8, f16: true, relu6: true }),
];
