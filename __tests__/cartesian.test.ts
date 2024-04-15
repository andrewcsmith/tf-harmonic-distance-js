import * as tf from "@tensorflow/tfjs-node"
import { expectTensorsClose } from "./test_utils"

import { 
    permutations
} from "../src/cartesian"

describe('permutations', () => {
    it('gets all 2-d permutations by default', async () => {
        const exp = tf.tensor([
            [[1, 0], [1, 0]],
            [[1, 0], [-1, 1]],
            [[-1, 1], [1, 0]],
            [[-1, 1], [-1, 1]]
        ])
        const res = permutations(tf.tensor([[1, 0], [-1, 1]]))
        await expectTensorsClose(res, exp)
    })

    it('gets all 1-d permutations', async () => {
        const exp = tf.tensor([
            [[1, 0]], [[-1, 1]]
        ])
        const res = permutations(tf.tensor([[1, 0], [-1, 1]]), 1)
        await expectTensorsClose(res, exp)
    })
})
