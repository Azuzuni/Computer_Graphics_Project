import * as THREE from 'three'
import { Object } from '../stage/Object';


export class Component {
  _object: Object | null = null;

  // Object refers to the classs Object inside stage directory
  constructor(object: Object) {
    this._object = object;
    if (this._object == null) console.log('Critical Error Component._object is null')
  }



}