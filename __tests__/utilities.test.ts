import { getBases } from "../src/utilities"
import * as tf from "@tensorflow/tfjs-node-gpu"

test('getBases gets bases', async () => {
    const res = await getBases(3)
    const exp = tf.tensor([[1, -0, -1], [-0, 1, 1]])
    expect(res.arraySync()).toEqual(exp.arraySync())
})
