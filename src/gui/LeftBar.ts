import GUI from 'lil-gui';
import * as THREE from 'three'
import * as UIFUNC from './UiFunctions'


const leftTab = new GUI({ 
  container: document.getElementsByName('left-window')[0],
  width: window.innerWidth * 0.10,
  title: '',
});

export function setUp() {
  UIFUNC.addButton( leftTab, "Add Box", ()=>{
    UIFUNC.addObject(new THREE.BoxGeometry(),new THREE.MeshLambertMaterial({ color: 0xFF0000}))
  });

}
