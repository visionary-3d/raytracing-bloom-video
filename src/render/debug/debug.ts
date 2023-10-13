// Purpose: Contains constants used throughout the application.

import { Value } from '../../utils/structs';
import { useGui } from '../init';

// constants
const GLOBAL_DEBUG = true;

export const DEBUG_INFO: DebugInfo = {
  BLOOM_COMPUTE_PASS: true,
  BLOOM_PREFILTER_PASS: true,
  BLOOM_DOWNSAMPLE_PASS: true,
  BLOOM_CLEAR_PASS: true,
  BLOOM_UPSAMPLE_PASS: true,
  BLOOM_TEXTURE_VIEW: false,
};

// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------

// logic

const DEBUG = new Value(GLOBAL_DEBUG);

export const initDebugInfo = () => {
  const values = Object.values(DEBUG_INFO);
  const keys = Object.keys(DEBUG_INFO);
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    const key = keys[i];

    DEBUG_INFO[key] = value && DEBUG.value;
  }
};

export const setupDebugGui = () => {
  const gui = useGui();

  const debugFolder = gui.addFolder('Debug Info');

  const keys = Object.keys(DEBUG_INFO);
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    debugFolder.add(DEBUG_INFO, key);
  }
};

type DebugInfo = { [key: string]: boolean };

export const useDebug = () => DEBUG.value;
