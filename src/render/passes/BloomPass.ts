import { Value } from '../../utils/structs';
import { DEBUG_INFO } from '../debug/debug';
import {
  Pass,
  Renderer,
  Uniform,
  UniformBuffer,
  useAspectRatioUniform,
  useGui,
  useResolutionUniform,
  useTimeUniform,
} from '../init';
import TimingHelper from '../libs/timing-helper';
import { Vector2 } from '../math/Vector2';
import BloomTextureViewer from '../utils/BloomTextureViewer';
import Encoder from './Encoder';
import bloomShader from './shaders/bloom.wgsl?raw';

const vec2_0 = new Vector2();

const uResolution = useResolutionUniform();
const uAspect = useAspectRatioUniform();
const uTime = useTimeUniform();

const uLerper = new Uniform(0);

const uBloomStrength = new Uniform(1);
const uBloomThreshold = new Uniform(4.5);
const uBloomSmoothness = new Uniform(3.5);

const uniforms = {
  uResolution,
  uAspect,
  uBloomStrength,
  uBloomThreshold,
  uBloomSmoothness,
  uLerper,
  uTime,
};

export const numMipLevels = (width: number, height: number) => {
  const maxSize = Math.max(width, height);
  return Math.log2(maxSize) | 0;
};

class BloomMip {
  size: Vector2;

  constructor(size: Vector2) {
    this.size = size;
  }
}

class DownsampleBindGroups {
  pingBindGroups: GPUBindGroup[];
  pongBindGroups: GPUBindGroup[];
  renderer: Renderer;
  uniformBuffer: UniformBuffer;
  ping: GPUTexture;
  pong: GPUTexture;
  bindGroupLayout: GPUBindGroupLayout;
  downsampleMipTexture: GPUTexture;

  constructor(
    ping: GPUTexture,
    pong: GPUTexture,
    downsampleMipTexture: GPUTexture,
    mipChainCount: number,
    renderer: Renderer,
    uniformBuffer: UniformBuffer,
    layout: GPUBindGroupLayout
  ) {
    this.renderer = renderer;
    this.uniformBuffer = uniformBuffer;
    this.ping = ping;
    this.pong = pong;
    this.downsampleMipTexture = downsampleMipTexture;
    this.bindGroupLayout = layout;

    const { pingBindGroups, pongBindGroups } =
      this.initBindGroups(mipChainCount);

    this.pingBindGroups = pingBindGroups;
    this.pongBindGroups = pongBindGroups;
  }

  initBindGroups(mipChainCount: number) {
    const device = this.renderer.device;

    const mipChainHalfCount = Math.floor(mipChainCount / 2);

    const pingBindGroups = new Array<GPUBindGroup>(mipChainHalfCount - 0);
    const pongBindGroups = new Array<GPUBindGroup>(mipChainHalfCount - 1);

    for (let i = 0; i < mipChainHalfCount; i++) {
      pingBindGroups[i] = device.createBindGroup({
        label: 'ping bind group descriptor bloom downsample',
        layout: this.bindGroupLayout,
        entries: [
          // width and height
          {
            binding: 0,
            resource: {
              buffer: this.uniformBuffer.buffer,
            },
          },
          {
            binding: 1,
            resource: this.ping.createView({
              baseMipLevel: i,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 2,
            resource: this.pong.createView({
              baseMipLevel: i,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 3,
            resource: this.downsampleMipTexture.createView({
              baseMipLevel: 0,
              mipLevelCount: 1,
            }),
          },
        ],
      } as GPUBindGroupDescriptor);
    }

    for (let i = 0; i < mipChainHalfCount - 1; i++) {
      pongBindGroups[i] = device.createBindGroup({
        label: 'pong bind group descriptor bloom downsample',
        layout: this.bindGroupLayout,
        entries: [
          // width and height
          {
            binding: 0,
            resource: {
              buffer: this.uniformBuffer.buffer,
            },
          },
          {
            binding: 1,
            resource: this.pong.createView({
              baseMipLevel: i + 1,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 2,
            resource: this.ping.createView({
              baseMipLevel: i,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 3,
            resource: this.downsampleMipTexture.createView({
              baseMipLevel: 0,
              mipLevelCount: 1,
            }),
          },
        ],
      } as GPUBindGroupDescriptor);
    }

    return { pingBindGroups, pongBindGroups };
  }
}

class UpsampleBindGroups {
  pingBindGroups: GPUBindGroup[];
  pongBindGroups: GPUBindGroup[];
  renderer: Renderer;
  uniformBuffer: UniformBuffer;
  ping: GPUTexture;
  pong: GPUTexture;
  bindGroupLayout: GPUBindGroupLayout;
  downsampleMipTexture: GPUTexture;

  constructor(
    ping: GPUTexture,
    pong: GPUTexture,
    downsampleMipTexture: GPUTexture,
    mipChainCount: number,
    renderer: Renderer,
    uniformBuffer: UniformBuffer,
    layout: GPUBindGroupLayout
  ) {
    this.renderer = renderer;
    this.uniformBuffer = uniformBuffer;
    this.ping = ping;
    this.pong = pong;
    this.downsampleMipTexture = downsampleMipTexture;
    this.bindGroupLayout = layout;

    const { pingBindGroups, pongBindGroups } =
      this.initBindGroups(mipChainCount);

    this.pingBindGroups = pingBindGroups;
    this.pongBindGroups = pongBindGroups;
  }

  initBindGroups(mipChainCount: number) {
    const device = this.renderer.device;

    const mipChainHalfCount = Math.floor(mipChainCount / 2);

    const pongBindGroups = new Array<GPUBindGroup>(mipChainHalfCount - 0);
    const pingBindGroups = new Array<GPUBindGroup>(mipChainHalfCount - 1);

    let downsampleIndex = 0;

    for (let i = 0; i < mipChainHalfCount; i++) {
      const index = mipChainHalfCount - 1 - i;
      const downsampleMipLevel = mipChainCount - downsampleIndex - 2;

      pongBindGroups[i] = device.createBindGroup({
        label: 'pong bind group descriptor bloom upsample',
        layout: this.bindGroupLayout,
        entries: [
          // width and height
          {
            binding: 0,
            resource: {
              buffer: this.uniformBuffer.buffer,
            },
          },
          {
            binding: 1,
            resource: this.pong.createView({
              baseMipLevel: index,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 2,
            resource: this.ping.createView({
              baseMipLevel: index,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 3,
            resource: this.downsampleMipTexture.createView({
              baseMipLevel: downsampleMipLevel,
              mipLevelCount: 1,
            }),
          },
        ],
      } as GPUBindGroupDescriptor);

      downsampleIndex += 2;
    }

    downsampleIndex = 1;

    for (let i = 0; i < mipChainHalfCount - 1; i++) {
      const index = mipChainHalfCount - 2 - i;
      const downsampleMipLevel = mipChainCount - downsampleIndex - 2;

      pingBindGroups[i] = device.createBindGroup({
        label: 'ping bind group descriptor bloom upsample',
        layout: this.bindGroupLayout,
        entries: [
          // width and height
          {
            binding: 0,
            resource: {
              buffer: this.uniformBuffer.buffer,
            },
          },
          {
            binding: 1,
            resource: this.ping.createView({
              baseMipLevel: index,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 2,
            resource: this.pong.createView({
              baseMipLevel: index + 1,
              mipLevelCount: 1,
            }),
          },
          {
            binding: 3,
            resource: this.downsampleMipTexture.createView({
              baseMipLevel: downsampleMipLevel,
              mipLevelCount: 1,
            }),
          },
        ],
      } as GPUBindGroupDescriptor);

      downsampleIndex += 2;
    }

    return { pingBindGroups, pongBindGroups };
  }
}

export class BloomPass extends Pass {
  downsamplePipeline: GPUComputePipeline;
  upsamplePipeline: GPUComputePipeline;
  ping: GPUTexture;
  pong: GPUTexture;
  downsampleBindGroups: DownsampleBindGroups;
  upsampleBindGroups: UpsampleBindGroups;
  source: GPUTexture;
  mipChainHalfCount: number;
  mips: BloomMip[];
  prefilterPipeline: GPUComputePipeline;
  prefilterBindGroup: GPUBindGroup;
  downsampleMipTexture: GPUTexture;
  bindGroupLayout: GPUBindGroupLayout;
  timerHelper: TimingHelper;
  prefilterTexture: GPUTexture;
  clearPipeline: GPUComputePipeline;
  clearBindGroup: GPUBindGroup;
  textureViewer: BloomTextureViewer;
  outputTexture: GPUTexture;
  LERPER_MAX: Value<number>;

  constructor(renderer: Renderer, outputTexture: GPUTexture) {
    super(renderer, bloomShader, uniforms);

    const resolutionVector = uResolution.value as Vector2;
    const width = resolutionVector.x;
    const height = resolutionVector.y;
    const mipChainCount = numMipLevels(width, height);
    this.mipChainHalfCount = Math.floor(mipChainCount / 2);

    this.ping = this.initOutputTexture(this.mipChainHalfCount);
    this.pong = this.initOutputTexture(this.mipChainHalfCount);
    this.outputTexture = this.pong;

    this.LERPER_MAX = new Value(400, 'bloom lerp samples');

    this.downsampleMipTexture = this.initInputTexture(mipChainCount - 1);
    this.prefilterTexture = this.initInputTexture(mipChainCount);
    this.source = outputTexture;

    this.timerHelper = new TimingHelper(renderer.device);

    this.mips = this.initMips();

    this.bindGroupLayout = this.initLayout();

    this.downsampleBindGroups = new DownsampleBindGroups(
      this.ping,
      this.pong,
      this.downsampleMipTexture,
      mipChainCount,
      renderer,
      this.uniformBuffer,
      this.bindGroupLayout
    );

    this.upsampleBindGroups = new UpsampleBindGroups(
      this.ping,
      this.pong,
      this.downsampleMipTexture,
      mipChainCount,
      renderer,
      this.uniformBuffer,
      this.bindGroupLayout
    );

    this.prefilterBindGroup = this.initStandardBindGroup(this.pong, this.ping);
    this.clearBindGroup = this.initStandardBindGroup(this.ping, this.pong);

    const {
      clearPipeline,
      prefilterPipeline,
      downsamplePipeline,
      upsamplePipeline,
    } = this.initPipelines();

    this.clearPipeline = clearPipeline;
    this.prefilterPipeline = prefilterPipeline;
    this.downsamplePipeline = downsamplePipeline;
    this.upsamplePipeline = upsamplePipeline;

    // debug
    this.textureViewer = new BloomTextureViewer(
      renderer,
      {
        mul: new Uniform(1),
      },
      this.prefilterTexture,
      this.ping,
      this.pong,
      this.downsampleMipTexture
    );

    // gui
    if (DEBUG_INFO.BLOOM_COMPUTE_PASS) {
      const gui = useGui();
      const bloomFolder = gui.addFolder('Bloom');
      bloomFolder.add(uBloomStrength, 'value', 0, 10, 0.001).name('strength');
      bloomFolder.add(uBloomThreshold, 'value', 0, 10, 0.001).name('threshold');
      bloomFolder
        .add(uBloomSmoothness, 'value', 0, 10, 0.001)
        .name('smoothness');
    }
  }

  initLayout() {
    const device = this.renderer.device;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: 'rgba16float',
            viewDimension: '2d',
          },
        },
        {
          binding: 2, // the texture buffer
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            viewDimension: '2d',
            sampleType: 'float',
            multisampled: false,
          },
        },
        {
          binding: 3, // the texture buffer
          visibility: GPUShaderStage.COMPUTE,
          texture: {
            viewDimension: '2d',
            sampleType: 'float',
            multisampled: false,
          },
        },
      ],
    } as GPUBindGroupLayoutDescriptor);

    return bindGroupLayout;
  }

  initMips() {
    const mipChainCount = this.mipChainHalfCount * 2;
    const resolutionVector = uResolution.value as Vector2;

    const mips = new Array<BloomMip>(mipChainCount);

    const prev = vec2_0.copy(resolutionVector);

    for (let i = 0; i < mipChainCount; i++) {
      mips[i] = new BloomMip(new Vector2().copy(prev));
      prev.set(Math.max(1, (prev.x / 2) | 0), Math.max(1, (prev.y / 2) | 0));
    }

    return mips;
  }

  initStandardBindGroup(output: GPUTexture, input: GPUTexture) {
    const device = this.renderer.device;

    const bindGroup = device.createBindGroup({
      label: 'prefilter bind group descriptor bloom',
      layout: this.bindGroupLayout,
      entries: [
        // width and height
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer.buffer,
          },
        },
        {
          binding: 1,
          resource: output.createView({
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
        {
          binding: 2,
          resource: input.createView({
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
        {
          binding: 3,
          resource: this.downsampleMipTexture.createView({
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
      ],
    } as GPUBindGroupDescriptor);

    return bindGroup;
  }

  initOutputTexture(mipChainCount: number = 1) {
    const device = this.renderer.device;

    const resolutionVector = uResolution.value as Vector2;

    const width = resolutionVector.x;
    const height = resolutionVector.y;

    const outputTexture = device.createTexture({
      label: 'output texture buffer',
      size: {
        width: width,
        height: height,
      },
      format: 'rgba16float',
      dimension: '2d',
      mipLevelCount: mipChainCount,
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    } as GPUTextureDescriptor);

    return outputTexture;
  }

  initInputTexture(mipChainCount: number = 1) {
    const device = this.renderer.device;

    const resolutionVector = uResolution.value as Vector2;

    const width = resolutionVector.x;
    const height = resolutionVector.y;

    const outputTexture = device.createTexture({
      label: 'bloom input texture buffer',
      size: {
        width: width,
        height: height,
      },
      format: 'rgba16float',
      dimension: '2d',
      mipLevelCount: mipChainCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    } as GPUTextureDescriptor);

    return outputTexture;
  }

  initPipelines() {
    const device = this.renderer.device;

    const shaderModule = device.createShaderModule({
      label: 'shader code',
      code: this.shader,
    });

    const prefilterPipeline = device.createComputePipeline({
      label: 'bloom prefilter compute pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'prefilter',
      },
    });

    const clearPipeline = device.createComputePipeline({
      label: 'bloom prefilter compute pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'clear',
      },
    });

    const downsampleDescriptor = {
      label: 'bloom downsample compute pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'down_sample',
      },
    };
    const downsamplePipeline =
      device.createComputePipeline(downsampleDescriptor);

    const upsamplePipeline = device.createComputePipeline({
      label: 'bloom upsample compute pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'up_sample',
      },
    });

    return {
      prefilterPipeline,
      downsamplePipeline,
      upsamplePipeline,
      clearPipeline,
    };
  }

  renderTextureViewer(debug: boolean = false) {
    if (debug) {
      this.textureViewer.render();
    }
  }

  dispatch(
    passEncoder: GPUComputePassEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    width: number,
    height: number
  ) {
    const workgroupCountX = Math.ceil(width / 8);
    const workgroupCountY = Math.ceil(height / 8);

    // Compute Pass
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  }

  getMipResolution(index: number) {
    return this.mips[index].size;
  }

  computePrefilter(encoder: Encoder, debug: boolean = false) {
    const passEncoder = encoder.getComputePassEncoder('bloom prefilter', debug);

    const resolutionVector = uResolution.value as Vector2;
    const width = resolutionVector.x;
    const height = resolutionVector.y;

    // output is now in pong
    this.dispatch(
      passEncoder,
      this.prefilterPipeline,
      this.prefilterBindGroup,
      width,
      height
    );

    passEncoder.end();

    const commandEncoder = encoder.getCommandEncoder();

    commandEncoder.copyTextureToTexture(
      { texture: this.pong },
      { texture: this.prefilterTexture },
      [width, height, 1]
    );
  }

  computeDownsample(encoder: Encoder, debug: boolean = false) {
    const passEncoder = encoder.getComputePassEncoder(
      'bloom downsample',
      debug
    );

    for (let i = 0; i < this.mipChainHalfCount; i++) {
      const index0 = (i + 0.5) * 2;
      const resolution = this.getMipResolution(index0);
      // console.log('first: ', resolution.x)

      // output is now in pong
      this.dispatch(
        passEncoder,
        this.downsamplePipeline,
        this.downsampleBindGroups.pingBindGroups[i],
        resolution.x,
        resolution.y
      );

      if (i < this.mipChainHalfCount - 1) {
        const index1 = (i + 1) * 2;
        const resolution = this.getMipResolution(index1);
        // console.log('second: ', resolution.x)

        // output is now in ping
        this.dispatch(
          passEncoder,
          this.downsamplePipeline,
          this.downsampleBindGroups.pongBindGroups[i],
          resolution.x,
          resolution.y
        );
      }
    }

    passEncoder.end();
  }

  saveDownsampledData(encoder: Encoder) {
    const commandEncoder = encoder.getCommandEncoder();

    for (let i = 0; i < this.mipChainHalfCount; i++) {
      const index0 = (i + 0.5) * 2;
      const resolution = this.getMipResolution(index0);

      const mipLevel = i * 2 + 0;

      commandEncoder.copyTextureToTexture(
        { texture: this.ping, mipLevel: i },
        { texture: this.downsampleMipTexture, mipLevel },
        [resolution.x, resolution.y, 1]
      );

      if (i < this.mipChainHalfCount - 1) {
        const index1 = (i + 1) * 2;
        const resolution = this.getMipResolution(index1);

        const mipLevel = i * 2 + 1;

        commandEncoder.copyTextureToTexture(
          { texture: this.pong, mipLevel: i + 1 },
          { texture: this.downsampleMipTexture, mipLevel },
          [resolution.x, resolution.y, 1]
        );
      }
    }
  }

  computeUpsample(encoder: Encoder, debug: boolean = false) {
    const passEncoder = encoder.getComputePassEncoder('bloom upsample', debug);

    const mipChainCount = this.mipChainHalfCount * 2;

    for (let i = 0; i < this.mipChainHalfCount; i++) {
      const index0 = mipChainCount - (i * 2 + 0) - 2;
      const resolution = this.getMipResolution(index0);

      // output is now in pong
      this.dispatch(
        passEncoder,
        this.upsamplePipeline,
        this.upsampleBindGroups.pongBindGroups[i],
        resolution.x,
        resolution.y
      );

      if (i < this.mipChainHalfCount - 1) {
        const index1 = mipChainCount - (i * 2 + 1) - 2;
        const resolution = this.getMipResolution(index1);

        // console.log(resolution.x)
        // output is now in ping
        this.dispatch(
          passEncoder,
          this.upsamplePipeline,
          this.upsampleBindGroups.pingBindGroups[i],
          resolution.x,
          resolution.y
        );
      }
    }

    passEncoder.end();
  }

  clearTexture(encoder: Encoder, debug: boolean = false) {
    const passEncoder = encoder.getComputePassEncoder(
      'bloom clear pass',
      debug
    );

    const resolutionVector = uResolution.value as Vector2;
    const width = resolutionVector.x;
    const height = resolutionVector.y;

    this.dispatch(
      passEncoder,
      this.clearPipeline,
      this.clearBindGroup,
      width,
      height
    );

    passEncoder.end();
  }

  prepareTextures(encoder: Encoder) {
    const commandEncoder = encoder.getCommandEncoder();

    const resolutionVector = uResolution.value as Vector2;
    const width = resolutionVector.x;
    const height = resolutionVector.y;

    commandEncoder.copyTextureToTexture(
      { texture: this.source },
      { texture: this.ping },
      [width, height, 1]
    );
  }

  compute(encoder: Encoder, frameValue: number, debug: boolean = false) {
    super.update(encoder, 0);

    uLerper.set(Math.min(frameValue / this.LERPER_MAX.value, 1));

    // store the texture in ping
    this.prepareTextures(encoder);

    // store the output in pong
    this.computePrefilter(encoder, DEBUG_INFO.BLOOM_PREFILTER_PASS && debug);

    // clear ping
    this.clearTexture(encoder, DEBUG_INFO.BLOOM_CLEAR_PASS && debug);

    // downsample and store the output in ping
    this.computeDownsample(encoder, DEBUG_INFO.BLOOM_DOWNSAMPLE_PASS && debug);

    // save downsampled data in a texture
    this.saveDownsampledData(encoder);

    // upsample and store the output in pong
    this.computeUpsample(encoder, DEBUG_INFO.BLOOM_UPSAMPLE_PASS && debug);

    // this.resetResolutionUniform(resolutionVector, savedResolution.x, savedResolution.y && debug);
    this.renderTextureViewer(DEBUG_INFO.BLOOM_TEXTURE_VIEW && debug);
  }
}
