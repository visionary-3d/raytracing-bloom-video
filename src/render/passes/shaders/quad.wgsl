struct ColorData {
  data: array<u32>,
};

struct Uniforms {
  resolution: vec2<f32>,
  aspect: f32,
  time: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var input_texture_sampler: sampler;

struct VertexOutput {
  @builtin(position) Position: vec4<f32>,
};

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
  var pos = array<vec2<f32>, 6> (
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(1.0, -1.0),
  vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.0, -1.0),
  vec2<f32>(-1.0, 1.0),
  vec2<f32>(1.0, 1.0)
  );

  var output: VertexOutput;
  output.Position = vec4<f32> (pos[VertexIndex], 0.0, 1.0);
  return output;
}

fn get_uvs(coord: vec3<f32>) -> vec2<f32> {
    var uv = vec2<f32>(coord.x, coord.y) / uniforms.resolution;

    // mirror y
    uv.y = 1.0 - uv.y;

    return uv;
}

const PI: f32 = 3.141592653589793;
const INV_PI: f32 = 0.31830988618379067153776752674503;
const INV_SQRT_OF_2PI: f32 = 0.39894228040143267793994605993439;

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let m1: mat3x3<f32> = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777)
    );
    let m2: mat3x3<f32> = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602)
    );
    let v: vec3<f32> = m1 * color;
    let a: vec3<f32> = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
    let b: vec3<f32> = v * (vec3<f32>(0.983729) * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
    return pow(clamp(m2 * (a / b), vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / 2.2));
}

fn denoise(tex: texture_2d<f32>, uv: vec2<f32>, sigma: f32, kSigma: f32, threshold: f32) -> vec4<f32> {
    let radius: f32 = round(kSigma * sigma);
    let radQ: f32 = radius * radius;

    let invSigmaQx2: f32 = 0.5 / (sigma * sigma);
    let invSigmaQx2PI: f32 = INV_PI * invSigmaQx2;

    let invThresholdSqx2: f32 = 0.5 / (threshold * threshold);
    let invThresholdSqrt2PI: f32 = INV_SQRT_OF_2PI / threshold;

    let centrPx: vec4<f32> = textureSample(tex, input_texture_sampler, uv);

    var zBuff: f32 = 0.0;
    var aBuff: vec4<f32> = vec4<f32>(0.0);
    let size: vec2<f32> = uniforms.resolution;

    for (var x: f32 = -radius; x <= radius; x = x + 1.0) {
        let pt: f32 = sqrt(radQ - x * x);
        for (var y: f32 = -pt; y <= pt; y = y + 1.0) {
            let d: vec2<f32> = vec2<f32>(x, y);

            let blurFactor: f32 = exp(-dot(d, d) * invSigmaQx2) * invSigmaQx2PI;

            let walkPx: vec4<f32> = textureSample(tex, input_texture_sampler, uv + d / size);

            let dC: vec4<f32> = walkPx - centrPx;
            let deltaFactor: f32 = exp(-dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;

            zBuff = zBuff + deltaFactor;
            aBuff = aBuff + deltaFactor * walkPx;
        }
    }
    return aBuff / zBuff;
}


@fragment
fn frag_main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {

  let uv = get_uvs(coord.xyz);

  // denoise
  let traced = denoise(input_texture, uv, 5.0, 1.0, 0.08).rgb;
  let bloom = textureSample(bloom_texture, input_texture_sampler, uv).rgb;
  var color = traced + bloom;

  // tonemap
  color = aces_tonemap(color);

  return vec4<f32>(color, 1.0);

}
