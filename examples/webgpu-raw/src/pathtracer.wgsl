// A tiny brute-force path tracer for a Cornell-box-ish scene, written in plain
// WGSL. One compute invocation per pixel adds ONE new path-traced sample and
// folds it into a running mean stored in a linear-HDR rgba16float texture
// (ping-ponged, so we can read last frame's mean while writing this frame's).
//
// It also writes the first-hit albedo and geometric normal into two more
// storage textures — the optional aux inputs the denoiser can consume. The
// whole scene is hardcoded below; there are no vertex buffers or acceleration
// structures. Naive path tracing is deliberately noisy at low sample counts —
// that noise is the point: it is what the denoiser cleans up.

struct Uniforms {
  res: vec2<u32>,
  frame: u32,       // 0-based sample index (also the RNG salt)
  auxEnabled: u32,  // write albedo/normal aux this frame (1) or not (0)
};

@group(0) @binding(0) var prevMean: texture_2d<f32>;                           // last frame's running mean
@group(0) @binding(1) var nextMean: texture_storage_2d<rgba16float, write>;    // this frame's running mean
@group(0) @binding(2) var albedoOut: texture_storage_2d<rgba16float, write>;   // aux: first-hit albedo [0,1] linear
@group(0) @binding(3) var normalOut: texture_storage_2d<rgba16float, write>;   // aux: first-hit normal [-1,1]
@group(0) @binding(4) var<uniform> u: Uniforms;

const PI = 3.14159265;
const MAX_BOUNCES = 6;
const EPS = 1e-3;

// ---- RNG (PCG hash) ---------------------------------------------------------
var<private> rngState: u32;
fn pcg() -> u32 {
  let s = rngState;
  rngState = s * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rand() -> f32 { return f32(pcg()) / 4294967296.0; }

// ---- geometry helpers -------------------------------------------------------
struct Hit {
  t: f32,
  pos: vec3<f32>,
  normal: vec3<f32>,
  albedo: vec3<f32>,
  emission: vec3<f32>,
  metal: f32,       // 1 = mirror-ish, 0 = diffuse
};

fn sphereT(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, r: f32) -> f32 {
  let oc = ro - center;
  let b = dot(oc, rd);
  let c = dot(oc, oc) - r * r;
  let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  let t = -b - sqrt(disc);
  return t;
}

// Scene: a box on [-1,1]^3, open at the front (+z) so the camera can look in.
// Left wall red, right wall green, everything else white; a bright quad on the
// ceiling is the only light. Two spheres inside: one mirror-ish, one diffuse.
fn intersect(ro: vec3<f32>, rd: vec3<f32>, hit: ptr<function, Hit>) -> bool {
  var tBest = 1e30;
  var found = false;

  // Axis-aligned walls. For each, solve the plane then bound the other two axes.
  // floor  y = -1  (n = +y, white)
  if (rd.y != 0.0) {
    let t = (-1.0 - ro.y) / rd.y;
    if (t > EPS && t < tBest) {
      let p = ro + t * rd;
      if (p.x >= -1.0 && p.x <= 1.0 && p.z >= -1.0 && p.z <= 1.0) {
        tBest = t; found = true;
        (*hit) = Hit(t, p, vec3(0.0, 1.0, 0.0), vec3(0.73), vec3(0.0), 0.0);
      }
    }
  }
  // ceiling y = +1 (n = -y) — white, except a central quad which is the light
  if (rd.y != 0.0) {
    let t = (1.0 - ro.y) / rd.y;
    if (t > EPS && t < tBest) {
      let p = ro + t * rd;
      if (p.x >= -1.0 && p.x <= 1.0 && p.z >= -1.0 && p.z <= 1.0) {
        tBest = t; found = true;
        let isLight = p.x > -0.4 && p.x < 0.4 && p.z > -0.5 && p.z < 0.3;
        let emis = select(vec3(0.0), vec3(20.0, 17.0, 13.0), isLight);
        (*hit) = Hit(t, p, vec3(0.0, -1.0, 0.0), vec3(0.73), emis, 0.0);
      }
    }
  }
  // back z = -1 (n = +z, white)
  if (rd.z != 0.0) {
    let t = (-1.0 - ro.z) / rd.z;
    if (t > EPS && t < tBest) {
      let p = ro + t * rd;
      if (p.x >= -1.0 && p.x <= 1.0 && p.y >= -1.0 && p.y <= 1.0) {
        tBest = t; found = true;
        (*hit) = Hit(t, p, vec3(0.0, 0.0, 1.0), vec3(0.73), vec3(0.0), 0.0);
      }
    }
  }
  // left x = -1 (n = +x, red)
  if (rd.x != 0.0) {
    let t = (-1.0 - ro.x) / rd.x;
    if (t > EPS && t < tBest) {
      let p = ro + t * rd;
      if (p.y >= -1.0 && p.y <= 1.0 && p.z >= -1.0 && p.z <= 1.0) {
        tBest = t; found = true;
        (*hit) = Hit(t, p, vec3(1.0, 0.0, 0.0), vec3(0.63, 0.06, 0.06), vec3(0.0), 0.0);
      }
    }
  }
  // right x = +1 (n = -x, green)
  if (rd.x != 0.0) {
    let t = (1.0 - ro.x) / rd.x;
    if (t > EPS && t < tBest) {
      let p = ro + t * rd;
      if (p.y >= -1.0 && p.y <= 1.0 && p.z >= -1.0 && p.z <= 1.0) {
        tBest = t; found = true;
        (*hit) = Hit(t, p, vec3(-1.0, 0.0, 0.0), vec3(0.14, 0.45, 0.16), vec3(0.0), 0.0);
      }
    }
  }

  // Mirror-ish sphere (glossy metal).
  let c0 = vec3(-0.42, -0.62, -0.28);
  let t0 = sphereT(ro, rd, c0, 0.38);
  if (t0 > EPS && t0 < tBest) {
    let p = ro + t0 * rd;
    tBest = t0; found = true;
    (*hit) = Hit(t0, p, normalize(p - c0), vec3(0.95), vec3(0.0), 1.0);
  }
  // Diffuse sphere (warm).
  let c1 = vec3(0.44, -0.66, 0.28);
  let t1 = sphereT(ro, rd, c1, 0.34);
  if (t1 > EPS && t1 < tBest) {
    let p = ro + t1 * rd;
    tBest = t1; found = true;
    (*hit) = Hit(t1, p, normalize(p - c1), vec3(0.78, 0.62, 0.26), vec3(0.0), 0.0);
  }
  return found;
}

// Cosine-weighted hemisphere sample around n.
fn cosineDir(n: vec3<f32>) -> vec3<f32> {
  let r1 = rand();
  let r2 = rand();
  let phi = 2.0 * PI * r1;
  let sinT = sqrt(r2);
  let cosT = sqrt(1.0 - r2);
  let a = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let tangent = normalize(cross(a, n));
  let bitan = cross(n, tangent);
  return normalize(tangent * (cos(phi) * sinT) + bitan * (sin(phi) * sinT) + n * cosT);
}
// Uniform point on the unit sphere (roughness perturbation for the metal).
fn sphereDir() -> vec3<f32> {
  let z = 1.0 - 2.0 * rand();
  let r = sqrt(max(0.0, 1.0 - z * z));
  let ph = 2.0 * PI * rand();
  return vec3(r * cos(ph), r * sin(ph), z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.res.x || gid.y >= u.res.y) { return; }
  let px = vec2<i32>(i32(gid.x), i32(gid.y));

  rngState = (gid.x * 1973u + gid.y * 9277u + u.frame * 26699u) | 1u;
  pcg(); pcg();

  // Camera: fixed, outside the open front face, looking down -z.
  let camPos = vec3(0.0, 0.0, 4.0);
  let tanHalfFov = 0.364; // ~40deg vertical
  let res = vec2<f32>(f32(u.res.x), f32(u.res.y));
  let jitter = vec2(rand(), rand());
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + jitter) / res;
  let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); // row 0 = top
  var rd = normalize(vec3(ndc.x * tanHalfFov, ndc.y * tanHalfFov, -1.0));
  var ro = camPos;

  var throughput = vec3(1.0);
  var radiance = vec3(0.0);
  var firstAlbedo = vec3(0.0);   // env default: albedo 0
  var firstNormal = vec3(0.0);   // env default: normal 0 (OIDN convention)

  for (var b = 0; b < MAX_BOUNCES; b = b + 1) {
    var h: Hit;
    if (!intersect(ro, rd, &h)) { break; } // miss -> dark room, no sky light
    if (b == 0) { firstAlbedo = h.albedo; firstNormal = h.normal; }
    radiance += throughput * h.emission;
    if (h.metal > 0.5) {
      rd = normalize(reflect(rd, h.normal) + 0.06 * sphereDir());
    } else {
      rd = cosineDir(h.normal);
    }
    throughput *= h.albedo;
    ro = h.pos + h.normal * EPS;
    // Russian roulette after a couple of bounces.
    if (b >= 2) {
      let p = max(throughput.r, max(throughput.g, throughput.b));
      if (rand() > p) { break; }
      throughput /= max(p, 1e-4);
    }
  }

  // Fold the new sample into the running mean: mean_n = mix(mean_{n-1}, s, 1/(n+1)).
  let prev = textureLoad(prevMean, px, 0).rgb;
  let n = f32(u.frame);
  let mean = mix(prev, radiance, 1.0 / (n + 1.0));
  textureStore(nextMean, px, vec4(mean, 1.0));

  if (u.auxEnabled == 1u) {
    textureStore(albedoOut, px, vec4(firstAlbedo, 1.0));
    textureStore(normalOut, px, vec4(firstNormal, 1.0));
  }
}
