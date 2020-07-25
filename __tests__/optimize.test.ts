import * as tf from "@tensorflow/tfjs-node-gpu"
import { expectTensorsClose } from "./test_utils"

import {
    parabolicLossFunction
} from "../src/optimize"
import { VectorSpace } from "../src/vectors";

describe('parabolicLossFunction', () => {
    describe('for one dimension', () => {
        let vs: VectorSpace

        beforeAll(async () => {
            vs = new VectorSpace({
                primeLimits: [2, 1], 
                bounds: [-1.0, 1.0]
            })
            await vs.init()
        })

        it('gets values at local minima', async () => {
            const exp = tf.tensor([1.0, 2.584962500, 0.0, 3.584962500, 1.0])
            const logPitches = tf.tensor([[1.0], [0.584962500], [0.0], [-0.4150375], [-1.0]])
            const curves = tf.fill([1], 0.01)
            const res = parabolicLossFunction(vs.pds, vs.hds, logPitches, curves)
            await expectTensorsClose(res, exp)
        })
    })

    describe('for two dimensions', () => {
        let vs: VectorSpace

        beforeAll(async () => {
            vs = new VectorSpace({
                primeLimits: [2, 1, 1], 
                bounds: [-1.0, 1.0],
                dimensions: 2,
            })
            await vs.init()
        })

        it('gets values at local minima', async () => {
            const logPitches = tf.tensor([
                [0.32192809, 0.5849625 ], // [5/4, 3/2]
                [0.26303441, 0.5849625 ], // [6/5, 3/2]
                [0.5849625 , 0.5849625 ]  // [3/2, 3/2]
            ])
            const exp = tf.tensor([5.9068906, 5.9068906, 2.5849625])
            const curves = tf.fill([2], 0.001)
            const res = parabolicLossFunction(vs.pds, vs.hds, logPitches, curves)
            await expectTensorsClose(res, exp)
        })
    })
})
