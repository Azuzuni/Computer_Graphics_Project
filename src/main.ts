import * as THREE from 'three'
import * as CORE from './core';
import * as EDITOR from './editor'
import * as GUI from './gui'
import * as STAGE from './stage'


function main() {
  CORE.setUp();
  EDITOR.setUp();
  GUI.setUp();
  STAGE.run();
}

main();