import * as THREE from 'three'
import * as CORE from '../core'
import {scene} from '../core'

const ambience = new THREE.AmbientLight(0xFFFFFF,1);
const light = new THREE.PointLight(0xFFFFFF, 100);
scene.add(ambience,light);
console.log('included')




function animate() {
  
  requestAnimationFrame( () => animate() );
}



export function run() {


    animate();
}