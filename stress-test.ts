import * as hdjs from './index'

let minimizer = new hdjs.Minimizer({
  primeLimits: [12, 12, 2, 1],
  dimensions: 2,
  bounds: [-4.0, 4.0],
  hdLimit: 9.0,
})

console.log("started!")

minimizer.vs.init().then(() => {
  console.log("initialized!")
})
