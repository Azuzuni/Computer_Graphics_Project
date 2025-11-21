import * as THREE from 'three';
import {scene,camera} from './Context';
import * as RAY from './Raycaster';


export const renderer = new THREE.WebGLRenderer();
// const viewWidth = window.innerWidth;
// const viewHeight = window.innerHeight;


// Use this function only once in main to SetUp variable parameters
// Additional calls to this function will just reset those values
export function setUpRenderer(support_lines: boolean=true) {
  renderer.setSize( window.innerWidth,  window.innerHeight );
  renderer.setClearColor( new THREE.Color( 0x333333 ) );
  document.getElementsByName('scene')[0].appendChild( renderer.domElement );
  renderer.setAnimationLoop(animate);
  if(!support_lines) return;
}

function animate () {
  // RAY.raycastFromCamera();
  // RAY.updateRayVisualization();
  renderer.render( scene, camera );
}

