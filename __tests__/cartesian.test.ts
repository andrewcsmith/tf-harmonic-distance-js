import * as tf from "@tensorflow/tfjs-node-gpu"
import { expectTensorsClose } from "./test_utils"

import { 
    permutations
} from "../src/cartesian"

test('permutations', async () => {
    const exp = tf.tensor([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]
    ])
    const res = permutations(tf.tensor([[0], [1]]))
    await expectTensorsClose(res, exp)
})
