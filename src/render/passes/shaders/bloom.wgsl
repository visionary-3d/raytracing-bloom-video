struct Uniforms {
  resolution: vec2<f32>,
  aspect: f32,
  strength: f32,
  threshold: f32,
  smoothness: f32,
  lerper: f32,
  time: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var input_texture: texture_2d<f32>;
@group(0) @binding(3) var downsample_texture: texture_2d<f32>;

const PI: f32 = 3.141592653589793;

fn is_outside_bounds(coord: vec2<u32>, bounds: vec2<f32>) -> bool {
    return coord.x >= u32(bounds.x) || coord.y >= u32(bounds.y);
}

fn get_uvs(coord: vec2<u32>) -> vec2<f32> {
    var uv = vec2<f32>(coord) / uniforms.resolution;

    // mirror y
    uv.y = 1.0 - uv.y;

    return uv;
}

fn get_coord(global_id: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(global_id.xy) / uniforms.resolution;
}

fn clip_coord(coord: vec2<u32>, min: vec2<u32>, max: vec2<u32>) -> vec2<u32> {
    return clamp(coord, min, max);
}

fn colorize(global_id: vec3<u32>, color: vec4<f32>) {
    textureStore(output_texture, global_id.xy, color);
}

// fn sample(texture: texture_2d<f32>, coord: vec2<u32>) -> vec4<f32> {
//     let new_coord = clip_coord(coord);
//     return textureLoad(texture, new_coord, 0);
// }

fn down_texture_sample(texture: texture_2d<f32>, coord: vec2<u32>) -> vec4<f32> {
    let new_coord = clip_coord(coord, vec2(0u), vec2<u32>(floor(uniforms.resolution)));
    return textureLoad(texture, new_coord, 0);
}

@compute @workgroup_size(8, 8) // 64
fn down_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // early return if we are out of bounds
    if is_outside_bounds(global_id.xy, floor(uniforms.resolution / 2.0)) {
        return;
    }

    let coord = global_id.xy * 2;

    let m1 = 1u;
    let m2 = 2u;

    let a = textureLoad(input_texture, vec2(coord.x - m2, coord.y + m2), 0).rgb;
    let b = textureLoad(input_texture, vec2(coord.x + 0u, coord.y + m2), 0).rgb;
    let c = textureLoad(input_texture, vec2(coord.x + m2, coord.y + m2), 0).rgb;

    let d = textureLoad(input_texture, vec2(coord.x - m2, coord.y + 0u), 0).rgb;
    let e = textureLoad(input_texture, vec2(coord.x + 0u, coord.y + 0u), 0).rgb;
    let f = textureLoad(input_texture, vec2(coord.x + m2, coord.y + 0u), 0).rgb;

    let g = textureLoad(input_texture, vec2(coord.x - m2, coord.y - m2), 0).rgb;
    let h = textureLoad(input_texture, vec2(coord.x + 0u, coord.y - m2), 0).rgb;
    let i = textureLoad(input_texture, vec2(coord.x + m2, coord.y - m2), 0).rgb;

    let j = textureLoad(input_texture, vec2(coord.x - m1, coord.y + m1), 0).rgb;
    let k = textureLoad(input_texture, vec2(coord.x + m1, coord.y + m1), 0).rgb;
    let l = textureLoad(input_texture, vec2(coord.x - m1, coord.y - m1), 0).rgb;
    let m = textureLoad(input_texture, vec2(coord.x + m1, coord.y - m1), 0).rgb;

    var downsample = e * 0.125;
    downsample += (a + c + g + i) * 0.03125;
    downsample += (b + d + f + h) * 0.0625;
    downsample += (j + k + l + m) * 0.125;

    colorize(global_id, vec4(downsample, 1.0));

    // let color = sample(global_id.xy, 0).rgb;
    // colorize(global_id, vec4<f32>(color, 1.0));
    
}

@compute @workgroup_size(8, 8) // 64
fn up_sample(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // let x = u32(uniforms.filter_radius);
    // let y = u32(uniforms.filter_radius);
    let x = 1u;
    let y = 1u;

    let coord = global_id.xy / 2;

    // Take 9 samples around current texel:
    // a - b - c
    // d - e - f
    // g - h - i
    // === ('e' is the current texel) ===

    let a = textureLoad(input_texture, vec2(coord.x - x, coord.y + y), 0).rgb;
    let b = textureLoad(input_texture, vec2(coord.x + 0, coord.y + y), 0).rgb;
    let c = textureLoad(input_texture, vec2(coord.x + x, coord.y + y), 0).rgb;
    let d = textureLoad(input_texture, vec2(coord.x - x, coord.y + 0), 0).rgb;
    let e = textureLoad(input_texture, vec2(coord.x + 0, coord.y + 0), 0).rgb;
    let f = textureLoad(input_texture, vec2(coord.x + x, coord.y + 0), 0).rgb;
    let g = textureLoad(input_texture, vec2(coord.x - x, coord.y - y), 0).rgb;
    let h = textureLoad(input_texture, vec2(coord.x + 0, coord.y - y), 0).rgb;
    let i = textureLoad(input_texture, vec2(coord.x + x, coord.y - y), 0).rgb;

    // Apply weighted distribution, by using a 3x3 tent filter:
    // | 1 2 1 |
    // | 2 4 2 |
    // | 1 2 1 |

    var upsample = e * 4.0;
    upsample += (b + d + f + h) * 2.0;
    upsample += (a + c + g + i);
    upsample /= 16.0;

    let prev = textureLoad(downsample_texture, coord, 0).rgb;

    colorize(global_id, vec4(prev + upsample, 1.0));
}

@compute @workgroup_size(8, 8) // 64
fn prefilter(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // early return if we are out of bounds
    if is_outside_bounds(global_id.xy, uniforms.resolution) {
        return;
    }

    let coord = global_id.xy;
    let color = textureLoad(input_texture, coord, 0).rgb;

    const MAGICAL_WEIGHTS: vec3<f32> = vec3(0.2126, 0.7152, 0.0722);
    let luminance = dot(color, MAGICAL_WEIGHTS);
    
    // smooth threshold
    let threshold = uniforms.threshold;
    let smoothness = uniforms.smoothness;
    let thresholded = vec3(smoothstep(threshold - smoothness, threshold + smoothness, luminance));
    let multiplier = thresholded * uniforms.strength * uniforms.lerper;

    colorize(global_id, vec4<f32>(color * multiplier, 1.0));

}

@compute @workgroup_size(8, 8) // 64
fn clear(@builtin(global_invocation_id) global_id: vec3<u32>) {

    colorize(global_id, vec4<f32>(0.0, 0.0, 0.0, 1.0));

}



