import { Object3D } from '../math/Object3D';
import { useCanvas, useRenderer } from './../init';
import { _calculateObjectSize } from './utils/objects';
import {
  clamp,
  lerp,
  easeOutExpo,
  EaseOutCirc,
  UpDownCirc,
  pow,
  easeUpDownExpo,
} from './utils/math';
import { Vector3 } from '../math/Vector3';
import { Quaternion } from '../math/Quaternion';
import { PerspectiveCamera } from '../math/Camera';

const HALF_PI = Math.PI / 2;
const FORWARD = new Vector3(0, 0, -1);
const LEFT = new Vector3(-1, 0, 0);
const UP = new Vector3(0, 1, 0);
const RIGHT = new Vector3(1, 0, 0);
const DOWN = new Vector3(0, -1, 0);

const quaternion_0 = new Quaternion();
const quaternion_1 = new Quaternion();
const vec3_0 = new Vector3();
const vec3_1 = new Vector3();
const vec3_2 = new Vector3();

const ZOOM_ANIMATION_PROGRESS_SPEED = 5;
const UP_DOWN_HEAD_ROTATION_LIMIT = Math.PI / 2;

const ONE = () => {
  return 1;
};
const TWO = () => {
  return 2;
};
const FIVE = () => {
  return 5;
};
const NEGATIVE_ONE = () => {
  return -1;
};
const ZERO = () => {
  return 0;
};

enum KEYS {
  // a = 'KeyA',
  // s = 'KeyS',
  // w = 'KeyD',
  // d = 'KeyF',
  a = 'KeyA',
  s = 'KeyS',
  w = 'KeyW',
  d = 'KeyD',
  space = 'Space',
  shiftL = 'ShiftLeft',
  shiftR = 'ShiftRight',
  enter = 'Enter',
}

const KEYS_LIST = Object.values(KEYS);

type KeyDown = {
  down: boolean;
  passedOneUpdateIteration: boolean;
};

type KeysDown = {
  [key: string]: KeyDown;
};

type NextKeysUp = {
  [key: string]: boolean;
};

type MouseState = {
  leftButton: boolean;
  rightButton: boolean;
  mouseXDelta: number;
  mouseYDelta: number;
  mouseX: number;
  mouseY: number;
  mouseWheelDirection: number;
  mouseWheelChanged: boolean;
};

class InputController {
  target: Document;
  currentMouse: MouseState;
  currentKeys: KeysDown;
  nextKeysUp: NextKeysUp;
  pointerLocked: boolean;
  lastTimestamp: number;

  constructor(target?: Document) {
    this.target = target || document;
    this.lastTimestamp = 0;
    this.currentMouse = {
      leftButton: false,
      rightButton: false,
      mouseXDelta: 0,
      mouseYDelta: 0,
      mouseX: 0,
      mouseY: 0,
      mouseWheelDirection: 0,
      mouseWheelChanged: false,
    };
    this.currentKeys = {};
    this.nextKeysUp = {};
    this.pointerLocked = false;
    this.init();
  }

  init() {
    const canvas = useCanvas();
    const keys = Object.values(KEYS);
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      // create supported keys objects
      this.currentKeys[key] = { down: false, passedOneUpdateIteration: false };
      this.nextKeysUp[key] = false;
    }

    this.target.addEventListener(
      'mousedown',
      (e) => this.onMouseDown(e),
      false
    );
    this.target.addEventListener(
      'mousemove',
      (e) => this.onMouseMove(e),
      false
    );
    this.target.addEventListener('mouseup', (e) => this.onMouseUp(e), false);
    this.target.addEventListener(
      'keydown',
      (e) => this.onKeyDown(e.code as KEYS),
      false
    );
    this.target.addEventListener(
      'keyup',
      (e) => this.onKeyUp(e.code as KEYS),
      false
    );
    addEventListener('wheel', (e) => this.onMouseWheel(e), false);

    const addPointerLockEvent = async () => {
      await canvas.requestPointerLock();
    };
    canvas.addEventListener('click', addPointerLockEvent);
    canvas.addEventListener('dblclick', addPointerLockEvent);
    canvas.addEventListener('mousedown', addPointerLockEvent);

    const setPointerLocked = () => {
      this.pointerLocked = document.pointerLockElement === canvas;
    };
    document.addEventListener('pointerlockchange', setPointerLocked, false);
  }

  onMouseWheel(e: WheelEvent) {
    const changeMouseWheelLevel = () => {
      if (this.pointerLocked) {
        if (e.deltaY < 0) {
          // console.log('scrolling up')
          // zooming in
          this.currentMouse.mouseWheelDirection = -1;
        } else if (e.deltaY > 0) {
          // console.log('scrolling down')
          this.currentMouse.mouseWheelDirection = 1;
        }

        this.currentMouse.mouseWheelChanged = true;
      }
    };

    changeMouseWheelLevel();
  }

  onMouseMove(e: MouseEvent) {
    if (this.pointerLocked) {
      this.currentMouse.mouseXDelta = e.movementX;
      this.currentMouse.mouseYDelta = e.movementY;
    }
  }

  onMouseDown(e: MouseEvent) {
    if (this.pointerLocked) {
      this.onMouseMove(e);

      switch (e.button) {
        case 0: {
          this.currentMouse.leftButton = true;
          break;
        }
        case 2: {
          this.currentMouse.rightButton = true;
          break;
        }
      }
    }
  }

  onMouseUp(e: MouseEvent) {
    if (this.pointerLocked) {
      this.onMouseMove(e);

      switch (e.button) {
        case 0: {
          this.currentMouse.leftButton = false;
          break;
        }
        case 2: {
          this.currentMouse.rightButton = false;
          break;
        }
      }
    }
  }

  onKeyDown(keyCode: KEYS) {
    if (this.pointerLocked && KEYS_LIST.includes(keyCode)) {
      this.currentKeys[keyCode].down = true;
      this.currentKeys[keyCode].passedOneUpdateIteration = false;
    }
  }

  onKeyUp(keyCode: KEYS) {
    if (this.pointerLocked && KEYS_LIST.includes(keyCode)) {
      // keyCode == KEYS.space && console.log(this.currentKeys[keyCode].down)

      const passed = this.currentKeys[keyCode].passedOneUpdateIteration;
      if (passed) {
        this.currentKeys[keyCode].down = false;
        this.currentKeys[keyCode].passedOneUpdateIteration = false;
      }

      this.nextKeysUp[keyCode] = !passed;
    }
  }

  hasKey(keyCode: string | number) {
    if (this.pointerLocked) {
      return this.currentKeys[keyCode].down;
    }

    return false;
  }

  update() {
    this.currentMouse.mouseXDelta = 0;
    this.currentMouse.mouseYDelta = 0;
    this.currentMouse.mouseWheelChanged = false;

    const currentKeysArray = Object.values(this.currentKeys);
    for (let i = 0; i < currentKeysArray.length; i++) {
      const key = currentKeysArray[i];
      if (key.down) {
        key.passedOneUpdateIteration = true;
      }
    }

    const nextKeysUpArrayKeys = Object.keys(this.nextKeysUp);
    const nextKeysUpArray = Object.values(this.nextKeysUp);
    for (let i = 0; i < nextKeysUpArray.length; i++) {
      const key = nextKeysUpArrayKeys[i];
      const up = nextKeysUpArray[i];
      if (up) {
        this.onKeyUp(key as KEYS);
      }
    }
  }

  runActionByKey(key: string, action: Function, inAction?: Function) {
    if (this.hasKey(key)) {
      return action();
    } else {
      return inAction && inAction();
    }
  }

  runActionByOneKey(
    keys: Array<string>,
    action: Function,
    inAction?: Function
  ) {
    let check = false;
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      check = this.hasKey(key);

      if (check) {
        break;
      }
    }

    if (check) {
      return action();
    } else {
      return inAction && inAction();
    }
  }

  runActionByAllKeys(
    keys: Array<string>,
    action: Function,
    inAction?: Function
  ) {
    let check = true;
    for (let i = 0; i < keys.length; i++) {
      const key = keys[i];
      check = this.hasKey(key);

      if (!check) {
        break;
      }
    }

    if (check) {
      return action();
    } else {
      return inAction && inAction();
    }
  }
}

class ZoomController {
  zoom: number;
  startZoomAnimation: number;
  isAnimating: boolean;

  constructor() {
    this.zoom = 0;
    this.startZoomAnimation = 0;
    this.isAnimating = false;
  }

  update(zoomChanged: boolean, zoomDirection: number, timestamp: number) {
    const time = timestamp;
    if (zoomChanged) {
      // restart the animation
      this.zoom = 0;
      this.startZoomAnimation = time;
      this.isAnimating = true;
    }

    // animating
    if (this.isAnimating) {
      const progress =
        (time - this.startZoomAnimation) * ZOOM_ANIMATION_PROGRESS_SPEED;

      if (progress >= 1) {
        // end the animation
        this.isAnimating = false;
        this.zoom = 0
      } else {
        this.zoom = easeUpDownExpo(progress) * zoomDirection;
      }
    }
  }
}

class CameraController extends Object3D {
  camera: PerspectiveCamera;
  inputController: InputController;
  movement: Vector3;
  phi: number;
  theta: number;
  objects: any;
  isMoving2D: boolean;
  zoomController: ZoomController;

  constructor(camera: PerspectiveCamera) {
    super();

    // init position
    this.position.copy(camera.position);
    this.quaternion.copy(camera.quaternion);

    // movement vector at every frame
    this.movement = new Vector3();

    this.camera = camera;

    this.inputController = new InputController();
    this.zoomController = new ZoomController();

    const q = this.quaternion;
    this.phi = 2 * Math.asin(-2.0 * (q.x * q.z - q.w * q.y));
    this.theta = Math.atan2(
      2.0 * (q.y * q.z + q.w * q.x),
      q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z
    );

    this.isMoving2D = false;
  }

  update(timestamp: number, timeDiff: number) {
    this.updateRotation();
    this.updateTranslation(timeDiff);
    this.zoomController.update(
      this.inputController.currentMouse.mouseWheelChanged,
      this.inputController.currentMouse.mouseWheelDirection,
      timestamp
    );
    this.updateCamera();
    this.inputController.update();
  }

  updateCamera() {
    this.camera.position.copy(this.position);
    this.camera.quaternion.copy(this.quaternion);
  }

  updateTranslation(timeDiff: number) {
    const time = timeDiff * 10;

    const shiftSpeedUpAction = () =>
      this.inputController.runActionByOneKey(
        [KEYS.shiftL, KEYS.shiftR],
        FIVE,
        ONE
      );

    const shiftNegativeAction = () =>
      this.inputController.runActionByKey(KEYS.enter, TWO, ZERO);

    const upwardVelocity =
      this.inputController.runActionByKey(KEYS.enter, ONE, ZERO) -
      this.inputController.runActionByOneKey(
        [KEYS.shiftL, KEYS.shiftR],
        shiftNegativeAction,
        ZERO
      );

    const forwardVelocity =
      this.inputController.runActionByKey(KEYS.w, shiftSpeedUpAction, ZERO) -
      this.inputController.runActionByKey(KEYS.s, shiftSpeedUpAction, ZERO);

    const sideVelocity =
      this.inputController.runActionByKey(KEYS.a, shiftSpeedUpAction, ZERO) -
      this.inputController.runActionByKey(KEYS.d, shiftSpeedUpAction, ZERO);

    // const qx = this.camera.quaternion
    const qx = quaternion_1;
    qx.setFromAxisAngle(UP, this.phi + HALF_PI);

    // Reset movement vector
    this.movement.set(0, 0, 0);

    const forwardMovement = vec3_0
      .copy(FORWARD)
      .applyQuaternion(this.camera.quaternion);
    forwardMovement.multiplyScalar(forwardVelocity * time);

    const leftMovement = vec3_1
      .copy(LEFT)
      .applyQuaternion(this.camera.quaternion);
    leftMovement.multiplyScalar(sideVelocity * time);

    const upwardMovement = vec3_2
      .copy(UP)
      .multiplyScalar(upwardVelocity * time);

    this.movement.add(forwardMovement);
    this.movement.add(leftMovement);
    this.movement.add(upwardMovement);

    this.isMoving2D = forwardVelocity != 0 || sideVelocity != 0;

    this.position.add(this.movement);

    const cameraDirection = this.camera.getWorldDirection(vec3_0);
    const zoom = this.zoomController.zoom;
    this.position.sub(cameraDirection.multiplyScalar(zoom));
  }

  updateRotation() {
    const xh =
      this.inputController.currentMouse.mouseXDelta / window.innerWidth;
    const yh =
      this.inputController.currentMouse.mouseYDelta / window.innerHeight;

    const PHI_SPEED = 2.5;
    const THETA_SPEED = 2.5;
    this.phi += -xh * PHI_SPEED;
    this.theta = clamp(
      this.theta + -yh * THETA_SPEED,
      -UP_DOWN_HEAD_ROTATION_LIMIT,
      UP_DOWN_HEAD_ROTATION_LIMIT
    );

    const qx = quaternion_0;
    qx.setFromAxisAngle(UP, this.phi);
    const qz = quaternion_1;
    qz.setFromAxisAngle(RIGHT, this.theta);

    const q = qx.multiply(qz);

    this.quaternion.copy(q);
  }
}

export default CameraController;
