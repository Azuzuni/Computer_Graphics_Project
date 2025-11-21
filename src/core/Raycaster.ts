import * as THREE from 'three';
import {scene,camera} from './Context';

const direction = new THREE.Vector3();
const raycaster = new THREE.Raycaster();
const rayArrow = new THREE.ArrowHelper(
  new THREE.Vector3(0, 0, -1), // default direction
  new THREE.Vector3(),         // origin
  5,                           // length
  0xff0000                     // color
);

export function raycastFromCamera() {
  camera.getWorldDirection(direction);
  raycaster.set(camera.position, direction);
  const hits = raycaster.intersectObjects(scene.children, true);
  scene.add(rayArrow);
  return hits; 
}

export function updateRayVisualization() {
    rayArrow.position.copy(camera.position);
    camera.getWorldDirection(direction);
    rayArrow.setDirection(direction);
    rayArrow.setLength(1);
}