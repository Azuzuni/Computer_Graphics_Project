import * as THREE from 'three'

export as namespace Object;
export = Object3D;
declare class Object3D  {
  _object: THREE.Mesh;
  constructor(geometry: THREE.BufferGeometry, material: THREE.Material);
}