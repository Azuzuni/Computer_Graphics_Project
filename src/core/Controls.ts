import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { camera } from './Context';
import { renderer } from './Renderer'
import { contain } from 'three/src/extras/TextureUtils.js';

const controls = new PointerLockControls( camera, renderer.domElement );


// ==========================================================
// Keyboard Controls
// ==========================================================

let keysPressed: Record<string,boolean> = {};


// Use this function only once in main to SetUp variable parameters
// Additional calls to this function will stack the input effect
export function setUpKeyboardControls() {
  const speed = 0.5;
  document.addEventListener('keydown', (event) => {
    keysPressed[event.key] = true;
    if ( keysPressed['w'] || keysPressed['ArrowUp']  )    camera.translateZ( -speed );
    if ( keysPressed['s'] || keysPressed['ArrowDown']  )  camera.translateZ(  speed );
    if ( keysPressed['a'] || keysPressed['ArrowLeft']  )  camera.translateX( -speed );
    if ( keysPressed['d'] || keysPressed['ArrowRight']  ) camera.translateX(  speed );
    if ( keysPressed[' '])     camera.translateY(  speed );
    if ( keysPressed['Shift']) camera.translateY( -speed );
    if ( keysPressed['Control'])  camera.translateY( -speed );
    controls.update(speed);
  });

  // Required to remove listener trigger when not intented
  document.addEventListener('keyup',   (event) => {keysPressed[event.key] = false});
}



// ==========================================================
// Mouse Controls
// ==========================================================

export function setUpMouseControls() {
  const container = document.getElementsByName('scene')[0];
  container.addEventListener('mousedown', onMouseDown);
  container.addEventListener('mouseup', onMouseUp);
}



function onMouseDown(event: MouseEvent) {
  controls.lock();
}

function onMouseUp(event: MouseEvent) {
  controls.unlock();
}