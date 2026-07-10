// Fullscreen tonemap blit for the RAW accumulation view (denoise toggle off).
// Reads the linear-HDR running-mean texture and writes ACES + sRGB to the
// canvas — the same display transform the denoiser applies with
// transfer:'aces-srgb', so toggling denoise on/off is a fair before/after.

@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  // One oversized triangle covering the viewport.
  let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  return vec4(p[i], 0.0, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}
fn toSrgb(x: vec3<f32>) -> vec3<f32> {
  let lo = x * 12.92;
  let hi = 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055;
  return select(hi, lo, x <= vec3(0.0031308));
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let c = textureLoad(src, vec2<i32>(pos.xy), 0).rgb;
  return vec4(toSrgb(aces(c)), 1.0);
}
