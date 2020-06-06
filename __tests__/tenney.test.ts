import { harmonicDistance } from "../src/tenney"
import * as tf from "@tensorflow/tfjs-node-gpu"

const FIFTH = tf.tensor([[-1, 1, 0]])
const TRIAD = tf.tensor([
    [-2, 0, 1],
    [-1, 1, 0]
])

test('harmonicDistance of one interval', async () => {
    const exp = tf.tensor([2.584962500721156])
    const res = harmonicDistance(FIFTH)
    expect(await res.sub(exp).sum().array()).toBeLessThan(1.0e-5)
})

test('harmonicDistance of multiple intervals', async () => {
    const exp = tf.tensor([4.321928094887363, 2.584962500721156])
    const res = harmonicDistance(TRIAD)
    expect(await res.sub(exp).sum().array()).toBeLessThan(1.0e-5)
})
