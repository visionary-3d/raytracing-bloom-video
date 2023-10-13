import { TickData } from './render/controllers/tick-manager';
import {
  useCameraUniform,
  useGpuDevice,
  useRenderer,
  useTick,
} from './render/init';

import Stats from './render/libs/Stats';

import { DEBUG_INFO, useDebug } from './render/debug/debug';
import { BloomPass } from './render/passes/BloomPass';
import Encoder from './render/passes/Encoder';
import { FullscreenPass } from './render/passes/FullscreenPass';
import { RaytracingPass } from './render/passes/RaytracingPass';
import { CameraStruct } from './render/utils/structs';

export const startApp = async () => {
  const renderer = useRenderer();
  const device = useGpuDevice();

  const raytracingPass = new RaytracingPass(renderer);

  const bloomPass = new BloomPass(renderer, raytracingPass.outputTexture);

  const fullscreenPass = new FullscreenPass(
    renderer,
    raytracingPass.outputTexture,
    bloomPass.outputTexture
  );

  const bloomStats = new Stats('Bloom').showPanel(2);
  const encoder = new Encoder(device, useDebug());

  useTick(({ timestamp }: TickData) => {
    bloomStats.begin();

    const sampling = raytracingPass.isSampling();

    if (sampling) {
      raytracingPass.update(encoder, timestamp);

      const frame = raytracingPass.getFrame();
      bloomPass.compute(encoder, frame, DEBUG_INFO.BLOOM_COMPUTE_PASS);

      fullscreenPass.update(encoder, timestamp);
      encoder.submit(bloomStats);
    }

    raytracingPass.updateUniforms();

    bloomStats.end();
  });
};
