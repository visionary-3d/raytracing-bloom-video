import { Object3D } from '../../math/Object3D';
import { Vector3 } from '../../math/Vector3';
import { Box3 } from '../../math/Box3';

const vec3_4 = new Vector3();

const _calculateObjectSize = (object: Object3D) => {
  const bbox = new Box3();
  bbox.expandByObject(object);
  const size = bbox.getSize(vec3_4);

  return size;
};

export { _calculateObjectSize };
