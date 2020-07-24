import * as tf from "@tensorflow/tfjs-node-gpu"

import { getBases } from "../src/utilities"
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