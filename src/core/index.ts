import * as CONTEXT from './Context'
import * as RENDERER from './Renderer'
import * as CONTROLS from './Controls'

export * from './Context'
export * from './Renderer'
export * from './Controls'

export function setUp() {
  CONTEXT.setUpVariables();
  RENDERER.setUpRenderer();
  CONTROLS.setUpKeyboardControls();
  CONTROLS.setUpMouseControls();
}