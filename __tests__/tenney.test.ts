import { 
    harmonicDistance, 
    harmonicDistanceAggregate 
} from "../src/tenney"

import * as tf from "@tensorflow/tfjs-node-gpu"

const FIFTH = tf.tensor([[-1, 1, 0]])
const TRIAD = tf.tensor([
    [-2, 0, 1],
    [-1, 1, 0]
])
const UNISON_FIFTH = tf.tensor([
    [0, 0, 0],
    [-1, 1, 0]
])

async function expectInRange(res: tf.Tensor, exp: tf.Tensor, range = 1.0e-5) {
    expect(await res.sub(exp).sum().array()).toBeLessThan(range)
}

test('harmonicDistance of one interval', async () => {
    const exp = tf.tensor([2.584962500721156])
    const res = harmonicDistance(FIFTH)
    expectInRange(res, exp)
})

test('harmonicDistance of multiple intervals', async () => {
    const exp = tf.tensor([4.321928094887363, 2.584962500721156])
    const res = harmonicDistance(TRIAD)
    expectInRange(res, exp)
})

test('harmonicDistance aggregate of UNISON_FIFTH', async () => {
    const exp = tf.tensor([2.584962500721156])
    const res = await harmonicDistanceAggregate(UNISON_FIFTH)
    expectInRange(res, exp)
})

test('harmonicDistance aggregate of TRIAD', async () => {
    const exp = tf.tensor([5.9068906])
    const res = await harmonicDistanceAggregate(TRIAD)
    expectInRange(res, exp)
})
