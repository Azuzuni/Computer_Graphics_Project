import GUI from 'lil-gui';
import * as THREE from 'three'
import * as UIFUNC from './UiFunctions'

const projectTab = new GUI({ 
  container: document.getElementsByName('top-bar')[0],
  width: window.innerWidth * 0.05,
  title: 'Project',
});

const windowsTab = new GUI({ 
  container: document.getElementsByName('top-bar')[0],
  width: window.innerWidth * 0.05,
  title: 'Windows',
});

const editorTab = new GUI({ 
  container: document.getElementsByName('top-bar')[0],
  width: window.innerWidth * 0.05,
  title: 'Editor',
});


export function setUp() {
  UIFUNC.addButton( projectTab, "Open Project", UIFUNC.openProject );
  UIFUNC.addButton( projectTab, "Save As", UIFUNC.saveAs );
  UIFUNC.addButton( projectTab, "Save", UIFUNC.save );
  UIFUNC.addButton( editorTab, "Settings", UIFUNC.save );
  


  projectTab.close();
  windowsTab.close();
  editorTab.close();
}


