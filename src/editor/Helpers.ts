import * as THREE from 'three';
import { scene, camera } from '../core/Context';


export function setUpHelpers() {
  // Grid size 100 divided into 30 chunks
  const gridHelper = new THREE.GridHelper(100,30);
  scene.add( gridHelper ); 
  
  // Axis size 5
  const axesHelper = new THREE.AxesHelper( 5 );
  scene.add( axesHelper );

}