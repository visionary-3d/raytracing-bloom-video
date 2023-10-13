struct ColorBuffer {
  values: array<u32>,
};

struct Camera { 
  position: vec4<f32>, 
  quaternion: vec4<f32>, 
  fov: f32, 
  near: f32, 
  far: f32, 
  tan_half_fov: f32,
};

struct Uniforms {
  resolution: vec2<f32>,
  aspect: f32,
  camera: Camera,
  frame: f32,
  time: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var previous_frames_texture: texture_2d<f32>;


struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

struct Material {
    color: vec3<f32>,
    emission: vec3<f32>,
    specular: vec3<f32>,

    emission_intensity: f32,
    specular_intensity: f32,
    roughness: f32,

    // metallic: f32,
    // ior: f32,
};

struct Hit {
    hit: bool,
    position: vec3<f32>,
    distance: f32,
    normal: vec3<f32>,
    material: Material,
};

struct Sphere {
    position: vec3<f32>,
    radius: f32,
    material: Material,
};

// Hash by David_Hoskins
const UI0 = 1597334673;
const UI1 = 3812015801;
const UI2 = vec2(UI0, UI1);
const UI3 = vec3(UI0, UI1, 2798796415);
const UIF = 1.0 / f32(0xffffffff);

fn hash33(p : vec3<f32>) -> vec3<f32> {
    var q : vec3<u32> = vec3<u32>(vec3(p)) * UI3;
    q = (q.x ^ q.y ^ q.z) * UI3;
    return -1.0 + 2.0 * vec3<f32>(q) * UIF;
}

fn remap(x : f32, a : f32, b : f32, c : f32, d : f32) -> f32 {
    return (((x - a) / (b - a)) * (d - c)) + c;
}

// Gradient noise by iq (modified to be tileable)
fn gradient_noise(x : vec3<f32>, freq : f32) -> f32 {
    // grid
    let p : vec3<f32> = floor(x);
    let w : vec3<f32> = fract(x);
    
    // quintic interpolant
    let u : vec3<f32> = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    
    // gradients
    let ga : vec3<f32> = hash33((p + vec3<f32>(0.0, 0.0, 0.0)) % freq);
    let gb : vec3<f32> = hash33((p + vec3<f32>(1.0, 0.0, 0.0)) % freq);
    let gc : vec3<f32> = hash33((p + vec3<f32>(0.0, 1.0, 0.0)) % freq);
    let gd : vec3<f32> = hash33((p + vec3<f32>(1.0, 1.0, 0.0)) % freq);
    let ge : vec3<f32> = hash33((p + vec3<f32>(0.0, 0.0, 1.0)) % freq);
    let gf : vec3<f32> = hash33((p + vec3<f32>(1.0, 0.0, 1.0)) % freq);
    let gg : vec3<f32> = hash33((p + vec3<f32>(0.0, 1.0, 1.0)) % freq);
    let gh : vec3<f32> = hash33((p + vec3<f32>(1.0, 1.0, 1.0)) % freq);
    
    // projections
    let va : f32 = dot(ga, w - vec3<f32>(0.0, 0.0, 0.0));
    let vb : f32 = dot(gb, w - vec3<f32>(1.0, 0.0, 0.0));
    let vc : f32 = dot(gc, w - vec3<f32>(0.0, 1.0, 0.0));
    let vd : f32 = dot(gd, w - vec3<f32>(1.0, 1.0, 0.0));
    let ve : f32 = dot(ge, w - vec3<f32>(0.0, 0.0, 1.0));
    let vf : f32 = dot(gf, w - vec3<f32>(1.0, 0.0, 1.0));
    let vg : f32 = dot(gg, w - vec3<f32>(0.0, 1.0, 1.0));
    let vh : f32 = dot(gh, w - vec3<f32>(1.0, 1.0, 1.0));
    
    // interpolation
    return va + 
           u.x * (vb - va) + 
           u.y * (vc - va) + 
           u.z * (ve - va) + 
           u.x * u.y * (va - vb - vc + vd) + 
           u.y * u.z * (va - vc - ve + vg) + 
           u.z * u.x * (va - vb - ve + vf) + 
           u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}

// Tileable 3D worley noise
fn worley_noise(uv : vec3<f32>, freq : f32) -> f32 {
    let id : vec3<f32> = floor(uv);
    let p : vec3<f32> = fract(uv);
    
    var minDist : f32 = 10000.0;
    for (var x : f32 = -1.0; x <= 1.0; x = x + 1.0) {
        for (var y : f32 = -1.0; y <= 1.0; y = y + 1.0) {
            for (var z : f32 = -1.0; z <= 1.0; z = z + 1.0) {
                let offset : vec3<f32> = vec3<f32>(x, y, z);
                var h : vec3<f32> = hash33((id + offset) % vec3<f32>(freq)) * 0.5 + 0.5;
                h = h + offset;
                let d : vec3<f32> = p - h;
                minDist = min(minDist, dot(d, d));
            }
        }
    }
    
    // inverted worley noise
    return 1.0 - minDist;
}

// Fbm for Perlin noise based on iq's blog
fn perling_fbm(p : vec3<f32>, freq : f32, octaves : i32) -> f32 {
    let G : f32 = exp2(-0.85);
    var amp : f32 = 1.0;
    var noise : f32 = 0.0;
    var f = freq;
    for (var i : i32 = 0; i < octaves; i = i + 1) {
        noise = noise + amp * gradient_noise(p * f, f);
        f = f * 2.0;
        amp = amp * G;
    }
    
    return noise;
}

// Tileable Worley fbm inspired by Andrew Schneider's Real-Time Volumetric Cloudscapes
// chapter in GPU Pro 7.
fn worley_fbm(p : vec3<f32>, freq : f32) -> f32 {
    return worley_noise(p * freq, freq) * 0.625 +
           worley_noise(p * freq * 2.0, freq * 2.0) * 0.25 +
           worley_noise(p * freq * 4.0, freq * 4.0) * 0.125;
}

//* Taken From Three.js
fn apply_quaternion(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {

  //calculate quat * vector
  var qv: vec4<f32> = vec4<f32> (
  q.w * v.x + q.y * v.z - q.z * v.y,
  q.w * v.y + q.z * v.x - q.x * v.z,
  q.w * v.z + q.x * v.y - q.y * v.x,
  -q.x * v.x - q.y * v.y - q.z * v.z
  );

  //calculate result * inverse quat
  return vec3<f32> (
  qv.x * q.w + qv.w * -q.x + qv.y * -q.z - qv.z * -q.y,
  qv.y * q.w + qv.w * -q.y + qv.z * -q.x - qv.x * -q.z,
  qv.z * q.w + qv.w * -q.z + qv.x * -q.y - qv.y * -q.x
  );
}

fn get_camera_to_pixel(coords: vec2<f32>) -> vec3<f32> {

  var camera = uniforms.camera;

  var uv = (coords - 0.5) * 2.0;
  uv.x *= uniforms.aspect;

  let d = 1.0 / uniforms.camera.tan_half_fov;
  let cameraToPixel = normalize(vec3(uv, -d));

  //* vector direction correction based on camera rotation
  let cameraToPixelRotated: vec3<f32> = apply_quaternion(cameraToPixel, camera.quaternion);

  //* direction of the vector
  let pixelViewDirection: vec3<f32> = normalize(cameraToPixelRotated);

  return pixelViewDirection;

}

fn get_index(coord: vec3<u32>) -> u32 {
    // return (coord.x + coord.y * u32(uniforms.size.x) + coord.z * u32(uniforms.size.x * uniforms.size.y)) * 3u;
    // return (coord.x + coord.y * u32(uniforms.size.x)) * 3u;
    return coord.x + coord.y * u32(uniforms.resolution.x);
}

fn colorize(global_id: vec3<u32>, color: vec4<f32>) {

    textureStore(output_texture, global_id.xy, color);

    // let index = get_index(global_id);

    // outputColorBuffer.values[index + 0u] = u32(r * 255f);
    // outputColorBuffer.values[index + 1u] = u32(g * 255f);
    // outputColorBuffer.values[index + 2u] = u32(b * 255f);
    // outputColorBuffer.values[index + 3u] = u32(a * 255f);
}

// * 2D hash function
fn hash2D(p: vec2<f32>) -> vec2<f32> {
    var c = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(c) * 18.5453);
}

fn mix(a: vec3<f32>, b: vec3<f32>, cond: bool) -> vec3<f32> {
    return a * (1.0 - f32(cond)) + b * f32(cond);
}

fn lerp1(a: f32, b: f32, cond: f32) -> f32 {
    return a * (1.0 - cond) + b * cond;
}

fn lerp2(a: vec2<f32>, b: vec2<f32>, cond: f32) -> vec2<f32> {
    return a * (1.0 - cond) + b * cond;
}

fn lerp3(a: vec3<f32>, b: vec3<f32>, cond: f32) -> vec3<f32> {
    return a * (1.0 - cond) + b * cond;
}

fn lerp4(a: vec4<f32>, b: vec4<f32>, cond: f32) -> vec4<f32> {
    return a * (1.0 - cond) + b * cond;
}

// return distance
fn voronoi(p: vec2<f32>) -> f32 {
    let n = floor(p);
    let f = fract(p);

    var m = vec3(8.0);

    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let g = vec2(f32(i), f32(j));
            let o = hash2D(n + g);
            let  r = g - f + (0.5 + 0.5 * sin(uniforms.time / 100.0 + 6.2831 * o));
            let d = dot(r, r);
            m = mix(m, vec3(d, o), d < m.x);
        }
    }

    return sqrt(m.x);
}

fn torus_sdf(p: vec3<f32>, t: vec2<f32>) -> f32 {
  let q: vec2<f32> = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}


fn is_outside_bounds(coord: vec2<u32>, bounds: vec2<f32>) -> bool {
    return coord.x >= u32(bounds.x) || coord.y >= u32(bounds.y);
}

fn get_uvs(coord: vec2<u32>) -> vec2<f32> {
    var uv = vec2<f32>(f32(coord.x) / uniforms.resolution.x, f32(coord.y) / uniforms.resolution.y);
    // var uv = vec2<f32>(f32(coord.x) / uniforms.resolution.x / uniforms.resolution.y);

    // make everything between 0 and 1
    // uv *= 2.0;

    // mirror y
    uv.y = 1.0 - uv.y;

    // take the aspect ratio into account
    // uv.x *= uniforms.aspec_ratio;

    return uv;
}


fn get_coord(global_id: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(global_id.xy) / uniforms.resolution;
}


const PI: f32 = 3.141592653589793;

// Classic Perlin 3D Noise
// by Stefan Gustavson
//
fn permute(x: vec4<f32>) -> vec4<f32> {
    return (((x * 34.0) + 1.0) * x) % 289.0;
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn fade(t: vec3<f32>) -> vec3<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn noise(P: vec3<f32>) -> f32 {
    var Pi0: vec3<f32> = floor(P); // Integer part for indexing
    var Pi1: vec3<f32> = Pi0 + vec3(1.0); // Integer part + 1
    Pi0 = Pi0 % 289.0;
    Pi1 = Pi1 % 289.0;
    let Pf0: vec3<f32> = fract(P); // Fractional part for interpolation
    let Pf1: vec3<f32> = Pf0 - vec3(1.0); // Fractional part - 1.0
    let ix: vec4<f32> = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    let iy: vec4<f32> = vec4(Pi0.yy, Pi1.yy);
    let iz0: vec4<f32> = Pi0.zzzz;
    let iz1: vec4<f32> = Pi1.zzzz;

    let ixy: vec4<f32> = permute(permute(ix) + iy);
    let ixy0: vec4<f32> = permute(ixy + iz0);
    let ixy1: vec4<f32> = permute(ixy + iz1);

    var gx0: vec4<f32> = ixy0 / 7.0;
    var gy0: vec4<f32> = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    let gz0: vec4<f32> = vec4(0.5) - abs(gx0) - abs(gy0);
    let sz0: vec4<f32> = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(vec4(0.0), gx0) - 0.5);
    gy0 -= sz0 * (step(vec4(0.0), gy0) - 0.5);

    var gx1: vec4<f32> = ixy1 / 7.0;
    var gy1: vec4<f32> = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    let gz1: vec4<f32> = vec4(0.5) - abs(gx1) - abs(gy1);
    let sz1: vec4<f32> = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(vec4(0.0), gx1) - 0.5);
    gy1 -= sz1 * (step(vec4(0.0), gy1) - 0.5);

    var g000: vec3<f32> = vec3(gx0.x, gy0.x, gz0.x);
    var g100: vec3<f32> = vec3(gx0.y, gy0.y, gz0.y);
    var g010: vec3<f32> = vec3(gx0.z, gy0.z, gz0.z);
    var g110: vec3<f32> = vec3(gx0.w, gy0.w, gz0.w);
    var g001: vec3<f32> = vec3(gx1.x, gy1.x, gz1.x);
    var g101: vec3<f32> = vec3(gx1.y, gy1.y, gz1.y);
    var g011: vec3<f32> = vec3(gx1.z, gy1.z, gz1.z);
    var g111: vec3<f32> = vec3(gx1.w, gy1.w, gz1.w);

    let norm0: vec4<f32> = taylor_inv_sqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    let norm1: vec4<f32> = taylor_inv_sqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;

    let n000: f32 = dot(g000, Pf0);
    let n100: f32 = dot(g100, vec3(Pf1.x, Pf0.yz));
    let n010: f32 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    let n110: f32 = dot(g110, vec3(Pf1.xy, Pf0.z));
    let n001: f32 = dot(g001, vec3(Pf0.xy, Pf1.z));
    let n101: f32 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    let n011: f32 = dot(g011, vec3(Pf0.x, Pf1.yz));
    let n111: f32 = dot(g111, Pf1);

    let fade_xyz: vec3<f32> = fade(Pf0);
    let n_z: vec4<f32> = lerp4(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    let n_yz: vec2<f32> = lerp2(n_z.xy, n_z.zw, fade_xyz.y);
    let n_xyz: f32 = lerp1(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

fn smooth_mod(axis: f32, amp: f32, rad: f32) -> f32 {
    let top: f32 = cos(PI * (axis / amp)) * sin(PI * (axis / amp));
    let bottom: f32 = pow(sin(PI * (axis / amp)), 2.0) + pow(rad, 2.0);
    let at: f32 = atan(top / bottom);
    return amp * (1.0 / 2.0) - (1.0 / PI) * at;
}

fn fit(unscaled: f32, original_min: f32, original_max: f32, min_allowed: f32, max_allowed: f32) -> f32 {
    return (max_allowed - min_allowed) * (unscaled - original_min) / (original_max - original_min) + min_allowed;
}

fn wave(position: vec3<f32>) -> f32 {
    return fit(smooth_mod(position.y * 6.0, 1.0, 1.5), 0.35, 0.6, 0.0, 1.0);
}

fn sample_sphere(position: vec3<f32>) -> f32 {
  var newPos = position;
  newPos.y += uniforms.time;
  let noisePattern = vec3<f32>(noise(newPos));
  return wave(noisePattern + uniforms.time);
}

fn sphere_sdf(p: vec3<f32>, s: f32) -> f32 {
  return length(p) - s;
}

fn shape_sdf(p: vec3<f32>, s: f32) -> f32 {
  let noise = sample_sphere(p * 1.3);
  let sdf = length(p + noise * normalize(p) * 10.0) - s;
  return lerp1(sdf, 1.0, f32(sdf <= 0.0));
}

fn calculate_normal(position: vec3<f32>) -> vec3<f32> {
  let eps: f32 = 0.001;
  let normal: vec3<f32> = vec3<f32>(
    shape_sdf(vec3<f32>(position.x + eps, position.y, position.z), 10.0) - shape_sdf(vec3<f32>(position.x - eps, position.y, position.z), 10.0),
    shape_sdf(vec3<f32>(position.x, position.y + eps, position.z), 10.0) - shape_sdf(vec3<f32>(position.x, position.y - eps, position.z), 10.0),
    shape_sdf(vec3<f32>(position.x, position.y, position.z + eps), 10.0) - shape_sdf(vec3<f32>(position.x, position.y, position.z - eps), 10.0)
  );
  return normalize(normal);
}

fn hash(p: vec3<f32>) -> f32 {
    return 0.5 + 0.5 * cos(sin(dot(floor(p + vec3<f32>(0.5)), vec3<f32>(113.1, 17.81, -33.58))) * 43758.545);
}

fn warp(p: vec3<f32>, n: i32) -> f32 {
    var v: f32 = 0.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        v = hash(p + vec3<f32>(v));
    }
    
    return v;
}

struct ColorStop {
    color: vec3<f32>,
    position: f32,
};

fn colorRamp(colors: array<ColorStop, 5>, factor: f32) -> vec3<f32> {
    var finalColor: vec3<f32> = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 5; i = i + 1) {
        var currentColor: ColorStop = colors[i];
        let isInBetween: bool = currentColor.position <= factor;
        finalColor = mix(finalColor, currentColor.color, isInBetween);
    }
    return finalColor;
}

fn glitch(id: f32, uv: vec2<f32>, range: vec2<f32>) -> vec3<f32> {
    const NUM_COLORS: i32 = 5;
    var colors: array<ColorStop, NUM_COLORS> = array<ColorStop, NUM_COLORS>(
        ColorStop(vec3<f32>(1.0, 1.0, 1.0), 0.0),
        ColorStop(vec3<f32>(1.0, 0.0, 0.0), 0.0),
        ColorStop(vec3<f32>(0.0, 1.0, 0.0), 0.0),
        ColorStop(vec3<f32>(0.0, 0.0, 1.0), 0.0),
        ColorStop(vec3<f32>(0.0, 0.0, 0.0), 0.0)
    );

    let INC: f32 = (range.y - range.x) / f32(NUM_COLORS - 2);

    // start
    colors[0].position = 0.0;

    // end
    colors[NUM_COLORS - 1].position = range.y;

    // rest
    for (var i: i32 = 1; i < 5 - 1; i = i + 1) {
        colors[i].position = range.x + f32(i - 1) * INC;
    }

    let s: f32 = clamp(smoothstep(range.x, range.y, uv.x), 0.0, 1.0);
    let selector: f32 = s + id * s;

    var color: vec3<f32> = colorRamp(colors, selector);

    return color;
}

fn rand(seed: ptr<function, u32>) -> f32 {
    (*seed) = (*seed) * 747796405u + 2891336453u;

    let new_seed = *seed;

    var result: u32 = ((new_seed >> ((new_seed >> 28u) + 4u)) ^ new_seed) * 277803737u;
    result = (result >> 22u) ^ result;

    return f32(result) / 4294967295.0;
}

// rand in normal distribution
fn rand_normal(seed: ptr<function, u32>) -> f32 {
    let theta = rand(seed) * 2.0 * PI;
    let r = sqrt(-2.0 * log(rand(seed)));
    return r * cos(theta);
}
 
// calculate random direction
fn rand_direction(seed: ptr<function, u32>) -> vec3<f32> {
    let x = rand_normal(seed);
    let y = rand_normal(seed);
    let z = rand_normal(seed);

    return normalize(vec3<f32>(x, y, z));
}

fn rand_hemi_direction(seed: ptr<function, u32>, normal: vec3<f32>) -> vec3<f32> {
    let dir = rand_direction(seed);
    return dir * sign(dot(dir, normal));
}

fn ray_sphere_intersection(ray: Ray, sphere: Sphere) -> Hit {
    let center: vec3<f32> = sphere.position;
    let radius: f32 = sphere.radius;

    let oc: vec3<f32> = ray.origin - center;
    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = 2.0 * dot(oc, ray.direction);
    let c: f32 = dot(oc, oc) - radius * radius;
    let discriminant: f32 = b * b - 4.0 * a * c;

    var hit: Hit = Hit(false, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), Material(vec3(0.0), vec3(0.0), vec3(1.0), 0.0, 0.0, 0.3));

    if (discriminant > 0.0) {
        let distance: f32 = (-b - sqrt(discriminant)) / (2.0 * a);
        if (distance > 0.0) {
            hit.hit = true;
            hit.position = ray.origin + ray.direction * distance;
            hit.distance = distance;
            hit.normal = normalize(hit.position - center);
            hit.material = sphere.material;
        }
    }

    return hit;
}

fn get_scene_hit(ray: Ray) -> Hit {

    let spheres_array: array<Sphere, 5> = array<Sphere, 5>(
        Sphere(vec3<f32>(3.0, 2.0, 0.0), 2.5, Material(vec3<f32>(1.0, 0.2, 0.0), vec3<f32>(0.0), vec3(1.0), 0.0, 0.5, 0.3)),
        Sphere(vec3<f32>(-3.0, 3.0, 0.0), 4.0, Material(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0), vec3(1.0), 0.0, 0.2, 0.0)),
        Sphere(vec3<f32>(0.1, -0.1, -1.0), 1.0, Material(vec3<f32>(1, 0, 0), vec3<f32>(1, 0, 0), vec3(1.0), 0.2, 0.0, 1.0)),
        Sphere(vec3<f32>(0.0, -20.0, 0.0), 20.0, Material(vec3<f32>(1.0), vec3<f32>(0.0), vec3(1.0), 0.1, 0.2, 0.2)),
        Sphere(vec3<f32>(0.0, 0.0, -1500.0), 100.0, Material(vec3<f32>(0.0), vec3<f32>(1.0), vec3(1.0), 2.0, 0.0, 0.3)),
    );

    var closest_hit = Hit(false, vec3<f32>(0.0), 1000000000.0, vec3<f32>(0.0), Material(vec3<f32>(0.0), vec3<f32>(0.0), vec3(1.0), 0.0, 0.0, 0.3));

    for(var i: i32 = 0; i < 5; i++) {
        let sphere = spheres_array[i];
        let hit = ray_sphere_intersection(ray, sphere);

        if(hit.hit && hit.distance < closest_hit.distance) {
            closest_hit = hit;
            closest_hit.material = sphere.material;
        }
    }

    return closest_hit;
}

fn shoot_ray(ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {

    const ambient_light: vec3<f32> = vec3<f32>(0.0002, 0.0003, 0.006);

    var new_ray = ray;
    var incoming_light = vec3(0.0);
    var ray_color = vec3(1.0);

    for(var i: u32 = 0u; i <= 7u; i++) {
        let hit = get_scene_hit(new_ray);

        if(hit.hit) {

            // origin
            new_ray.origin = hit.position;

            // direction

            // let diffuse_direction = rand_hemi_direction(seed, hit.normal);
            let diffuse_direction = normalize(hit.normal + rand_hemi_direction(seed, hit.normal));
            let specular_direction = reflect(new_ray.direction, hit.normal);
            let is_specular = rand(seed) < hit.material.specular_intensity;

            new_ray.direction = normalize(lerp3(diffuse_direction, 
                                                specular_direction, 
                                                hit.material.roughness * f32(is_specular)));

            // based on roughness
            let roughness = hit.material.roughness;
            // new_ray.direction = normalize(lerp3(hit.normal, rand_hemi_direction(seed, hit.normal), roughness));


            let material = hit.material;

            let emitted_light = material.emission * material.emission_intensity;
            incoming_light += emitted_light * ray_color;
            ray_color *= lerp3(material.color, material.specular, f32(is_specular));

        } else {
            
            incoming_light += ambient_light * ray_color;
            break;
        }
    }


    return incoming_light;

}

@compute @workgroup_size(8, 8) // 64
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // early return if we are out of bounds
    if is_outside_bounds(global_id.xy, uniforms.resolution) {
        return;
    }

    let uv = get_uvs(global_id.xy);

    let pixel_coord = get_coord(global_id.xy);

    // Define the origin of the ray
    let ray_origin: vec3<f32> = uniforms.camera.position.xyz;

    // Define the direction of the ray
    let ray_direction: vec3<f32> = get_camera_to_pixel(pixel_coord);

    let ray: Ray = Ray(ray_origin, ray_direction);

    var seed = get_index(global_id) + u32(uniforms.frame) * 7572u;
    var color = vec3(0.0);

    const trace_per_pixel: u32 = 50u;
    const color_divider: f32 = f32(trace_per_pixel) / 100.0;
    for (var i: u32 = 0u; i < trace_per_pixel; i++) {
        color += shoot_ray(ray, &seed) / color_divider;
    }

    let old = textureLoad(previous_frames_texture, vec2<i32>(global_id.xy), 0).rgb;
    let weight = 1.0 / (uniforms.frame);

    color = lerp3(old, color, weight);

    // store it in the texture
    colorize(global_id, vec4<f32>(color, 1.0));
    
}