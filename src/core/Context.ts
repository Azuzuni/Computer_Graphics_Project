import * as THREE from 'three';


export const scene = new THREE.Scene();
export const camera = new THREE.PerspectiveCamera( 75, window.innerWidth/window.innerHeight, 0.1, 1000 );



// Use this function only once in main to SetUp variable parameters
// Additional calls to this function will just reset those values
export function  setUpVariables() {
  camera.position.z = 5;
  camera.position.y = 5;
  camera.lookAt( scene.position );
}
