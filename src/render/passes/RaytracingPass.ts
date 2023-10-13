import { Value } from '../../utils/structs';
import {
  Pass,
  Renderer,
  Uniform,
  useAspectRatioUniform,
  useCamera,
  useCameraUniform,
  useResolutionUniform,
  useTimeUniform,
} from '../init';
import { Vector2 } from '../math/Vector2';
import Encoder from '../passes/Encoder';
import { numMipLevels } from './BloomPass';
import raytraceShader from './shaders/raytrace.wgsl?raw';

const uResolution = useResolutionUniform();
const uAspect = useAspectRatioUniform();
const uTime = useTimeUniform();
const uCamera = useCameraUniform();
const uFrame = new Uniform(1);

const uniforms = {
  uResolution,
  uAspect,
  uCamera,
  uFrame,
  uTime,
};

export class RaytracingPass extends Pass {
  pipeline: GPUComputePipeline;
  inputTexture: GPUTexture;
  outputTexture: GPUTexture;
  bindGroupLayout: GPUBindGroupLayout;
  bindGroupDescriptor: GPUBindGroupDescriptor;
  bindGroup: GPUBindGroup;
  postSubmitIsSet: boolean;
  oldCamera: import('/home/arya/Desktop/visionary/compute-power/src/render/math/Camera').PerspectiveCamera;
  MAX_SAMPLES: Value<number>;

  constructor(renderer: Renderer) {
    super(renderer, raytraceShader, uniforms);

    this.inputTexture = this.initInputTexture();
    this.outputTexture = this.initOutputTexture();

    this.bindGroupLayout = this.initLayout();
    this.bindGroupDescriptor = this.initDescriptor();

    this.bindGroup = this.initBindGroup();

    this.pipeline = this.initPipeline();

    const camera = useCamera();
    this.oldCamera = camera.clone();

    this.postSubmitIsSet = false;
    this.MAX_SAMPLES = new Value(1000, 'samples');
  }

  updateUniforms() {
    const camera = useCamera();

    if (this.oldCamera.equals(camera) === false) {
      // reset, if camera has changed
      this.oldCamera.copy(camera);
      uFrame.set(1);
    }

    if(this.isSampling()) {
      uFrame.set((uFrame.value as number) + 1);
    }
  }

  initBindGroup() {
    const device = this.renderer.device;
    const bindGroup = device.createBindGroup(this.bindGroupDescriptor);
    return bindGroup;
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
          binding: 2, // the old texture buffer
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

  initPipeline() {
    const device = this.renderer.device;

    const shaderModule = device.createShaderModule({
      label: 'shader code',
      code: this.shader,
    });

    const computePipeline = device.createComputePipeline({
      label: 'raytrace compute pipeline',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    return computePipeline;
  }

  initOutputTexture() {
    const device = this.renderer.device;

    const resolutionVector = uResolution.value as Vector2;

    const width = resolutionVector.x;
    const height = resolutionVector.y;

    const mipChainCount = numMipLevels(width, height);

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

  initInputTexture() {
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
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    } as GPUTextureDescriptor);

    return outputTexture;
  }

  initDescriptor() {
    const bindGroupDescriptor = {
      label: 'bind group descriptor',
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
          resource: this.outputTexture.createView({
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
        {
          binding: 2,
          resource: this.inputTexture.createView(),
        },
      ],
    } as GPUBindGroupDescriptor;

    return bindGroupDescriptor;
  }

  compute(encoder: Encoder, debug: boolean = false) {
    const passEncoder = encoder.getComputePassEncoder('raytracing', debug);

    const resolutionVector = uResolution.value as Vector2;

    const width = resolutionVector.x;
    const height = resolutionVector.y;

    const workgroupCountX = Math.ceil(width / 8);
    const workgroupCountY = Math.ceil(height / 8);

    // Compute Pass
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);

    passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    passEncoder.end();
  }

  getFrame() {
    return uFrame.value as number;
  }

  isSampling() {
    const frameValue = uFrame.value as number;
    const sampling = frameValue <= this.MAX_SAMPLES.value;

    return sampling;
  }

  update(encoder: Encoder, timestamp: number, debug: boolean = false) {

    super.update(encoder, timestamp, debug);
    this.compute(encoder, debug);
    this.updateOldTexture(encoder);
  }

  updateOldTexture(encoder: Encoder) {
    const commandEncoder = encoder.getCommandEncoder();
    const resolutionVector = uResolution.value as Vector2;

    const width = resolutionVector.x;
    const height = resolutionVector.y;

    commandEncoder.copyTextureToTexture(
      { texture: this.outputTexture },
      { texture: this.inputTexture },
      [width, height, 1]
    );
  }
}
