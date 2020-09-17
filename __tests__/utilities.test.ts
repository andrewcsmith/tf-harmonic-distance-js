import * as tf from "@tensorflow/tfjs-node-gpu"

import { 
    getBases,
    transformToUnitCircle,
    inverseTransformToUnitCircle,
} from "../src/utilities"

import { expectTensorsClose } from "./test_utils"

describe('getBases', () => {
    it('gets 2-combos', async () => {
        const res = await getBases(2)
        const exp = tf.tensor([[1]])
        await expectTensorsClose(res, exp)
    })

    it('gets 3-combos', async () => {
        const res = await getBases(3)
        const exp = tf.tensor([[1, -0, -1], [-0, 1, 1]])
        await expectTensorsClose(res, exp)
    })
}) 

describe('transformToUnitCircle', () => {
    const SQRT_OF_HALF = Math.sqrt(0.5)

    it('properly transforms a 1d space', async () => {
        const pds = tf.tensor([
            [0.0],
            [1.0],
            [0.5],
        ])
        const exp = [
            [0.0],
            [1.0],
            [0.5],
        ]
        const res = await transformToUnitCircle(pds)
        await expectTensorsClose(res, tf.tensor(exp))
    })

    it('properly transforms a 2d space', async () => {
        const pds = tf.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        const exp = [
            [0.0, 0.0],
            [SQRT_OF_HALF, SQRT_OF_HALF],
            [0.0, 1.0], 
            [1.0, 0.0],
            [SQRT_OF_HALF*0.5, SQRT_OF_HALF*0.5],
            [0.5, 0.0],
            [0.0, 0.5]
        ]
        const res = transformToUnitCircle(pds)
        await expectTensorsClose(res, tf.tensor(exp))
    })
})

describe('inverseTransformToUnitCircle', () => {
    const SQRT_OF_HALF = Math.sqrt(0.5)

    it('properly transforms a 1d space', async () => {
        const pds = tf.tensor([
            [0.0],
            [1.0],
            [0.5],
        ])
        const exp = tf.tensor([
            [0.0],
            [1.0],
            [0.5],
        ])
        const res = inverseTransformToUnitCircle(pds)
        await expectTensorsClose(res, exp)
    })

    it('properly transforms a 2d space', async () => {
        const pds = tf.tensor([
            [0.0, 0.0],
            [SQRT_OF_HALF, SQRT_OF_HALF],
            [0.0, 1.0], 
            [1.0, 0.0],
            [SQRT_OF_HALF*0.5, SQRT_OF_HALF*0.5],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        const exp = tf.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        const res = await inverseTransformToUnitCircle(pds)
        await expectTensorsClose(res, exp)
    })
})