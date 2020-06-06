import { pitchDistance } from "../src/vectors"
import * as tf from "@tensorflow/tfjs-node-gpu"

test('pitchDistance for one vec', () => {
    const v = tf.tensor([[-1, 1, 0]])
    const exp = 0.58
    const res = pitchDistance(v)
    expect(res.arraySync()[0]).toBeCloseTo(exp)
})

test('pitchDistance for multiple vecs', () => {
    const v = tf.tensor([[-1, 1, 0], [-2, 0, 1]])
    const exp = [0.5850, 0.3219]
    const res = pitchDistance(v)
    expect(res.arraySync()[0]).toBeCloseTo(exp[0])
    expect(res.arraySync()[1]).toBeCloseTo(exp[1])
})
