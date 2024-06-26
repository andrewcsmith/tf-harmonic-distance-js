const hdjs = require('hd-js')
const tf = require('@tensorflow/tfjs-node')

const callback = async function() {
  let pitches = await minimizer.logPitches.array()
  console.log(pitches[0])
}

let minimizer = new hdjs.Minimizer({
  primeLimits: [19, 12, 2, 2, 1, 1],
  dimensions: 3,
  bounds: [-4.0, 4.0],
  hdLimit: 9.0,
  callback: callback,
})

console.log("Loaded the script")

minimizer.vs.init().then(() => {
  console.log("Initialized minimizer VectorSpace")
}).catch(err => {
  console.log(err)
})