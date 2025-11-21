import * as THREE from 'three'
import { scene } from '../core'
import Object3D = require("@object")

export class Object3D {
  _object: THREE.Mesh | null = null;
  
  constructor(geometry: THREE.BufferGeometry, material: THREE.Material) {
    this._object = new THREE.Mesh(geometry,material);
    scene.add(this._object);
  }
}