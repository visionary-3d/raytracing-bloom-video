import Encoder from './Encoder';
import { Pass, Renderer, UniformList, useAspectRatioUniform, useResolutionUniform, useTimeUniform } from '../init';
import quadShader from './shaders/quad.wgsl?raw';

const uResolution = useResolutionUniform();
const uAspect = useAspectRatioUniform();
const uTime = useTimeUniform()

const uniforms = {
  uResolution,
  uAspect,
  uTime,
};

export class FullscreenPass extends Pass {
  bindGroupLayout: GPUBindGroupLayout;
  renderPipeline: GPURenderPipeline;
  bindGroup: GPUBindGroup;
  renderPassDescriptor: any;

  constructor(
    renderer: Renderer,
    outputTexture: GPUTexture,
    bloomTexture: GPUTexture
  ) {
    super(renderer, quadShader, uniforms);

    const device = this.renderer.device;

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 1, // the texture buffer
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            viewDimension: '2d',
            sampleType: 'float',
            multisampled: false,
          },
        },
        {
          binding: 2, // the texture buffer
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            viewDimension: '2d',
            sampleType: 'float',
            multisampled: false,
          },
        },
        {
          binding: 3, // the sampler
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {
            type: 'filtering',
          },
        },
      ],
    });

    this.renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout],
      }),
      vertex: {
        module: device.createShaderModule({
          code: this.shader,
        }),
        entryPoint: 'vert_main',
      },
      fragment: {
        module: device.createShaderModule({
          code: this.shader,
        }),
        entryPoint: 'frag_main',
        targets: [
          {
            format: this.renderer.presentationFormat as GPUTextureFormat,
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });

    const sampler = device.createSampler({
      label: 'sampler huh',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      addressModeW: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
      // mipmapFilter: 'linear',
    });

    const bindGroupDescriptor = {
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer.buffer,
          },
        },
        {
          binding: 1,
          resource: outputTexture.createView(),
        },
        {
          binding: 2,
          resource: bloomTexture.createView({
            baseMipLevel: 0,
            mipLevelCount: 1,
          }),
        },
        {
          binding: 3,
          resource: sampler,
        },
      ],
    };

    this.bindGroup = device.createBindGroup(bindGroupDescriptor);

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: this.renderer.context.getCurrentTexture().createView(), // Assigned later

          clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    } as any;
  }

  render = (encoder: Encoder, debug: boolean = false) => {
    this.renderPassDescriptor.colorAttachments[0].view = this.renderer.context
      .getCurrentTexture()
      .createView();

    const passEncoder = encoder.getRenderPassEncoder(
      'quad shader pass',
      this.renderPassDescriptor,
      debug
    );
    passEncoder.setPipeline(this.renderPipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
  };

  update(encoder: Encoder, timestamp: number, debug: boolean = false) {
    super.update(encoder, timestamp, debug);
    this.render(encoder, debug);
  }
}
