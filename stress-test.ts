import * as hdjs from './index'

let minimizer = new hdjs.Minimizer({
  primeLimits: [19, 12, 2, 2, 1, 1],
  dimensions: 3,
  bounds: [-4.0, 4.0],
  hdLimit: 9.0,
})

console.log("started!")

minimizer.vs.init().then(() => {
  console.log("initialized!")
})