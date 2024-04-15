import * as Benchmark from "benchmark"
import * as tf from "@tensorflow/tfjs-node"

import { 
    VectorSpace,
} from "../src/vectors"

var suite = new Benchmark.Suite

suite.add("cool test", async () => {
    let vs = new VectorSpace({
        primeLimits: [3, 2],
        dimensions: 2,
        hdLimit: 9.0,
        bounds: [-3.0, 3.0]
    })
    await vs.init()
})
.on('cycle', event => {
    const benchmark = event.target
    console.log(benchmark.toString())
})
.run({ async: true })
