import GUI from 'lil-gui';
import * as THREE from 'three'
import { Object } from '../stage/Object'


export function addButton(gui: GUI, label: string, func: () => void) {
  const obj: Record<string, () => void> = {
    [label]: func
  };

  gui.add(obj, label);
}



export function openProject() {
  alert( 'Open Project is WIP' );
}

export function saveAs() {
  alert( 'Save As is WIP' );
}


export function save() {
  alert( 'Save is WIP' );
}


export function addObject(geometry: THREE.BufferGeometry, material: THREE.Material) {
 return new Object(geometry,material);
}