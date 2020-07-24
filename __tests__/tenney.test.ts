import * as tf from "@tensorflow/tfjs-node-gpu"
import { expectTensorsClose } from "./test_utils"

import { 
    harmonicDistance, 
    harmonicDistanceAggregate 
} from "../src/tenney"

const FIFTH = tf.tensor([[-1, 1, 0]])
const TRIAD = tf.tensor([[
    [-2, 0, 1],
    [-1, 1, 0]
]])
const UNISON_FIFTH = tf.tensor([[
    [0, 0, 0],
    [-1, 1, 0]
]])

describe('harmonicDistance', () => {
    it('evaluates one interval', async () => {
        const exp = tf.tensor([2.584962500721156])
        const res = harmonicDistance(FIFTH)
        await expectTensorsClose(res, exp)
    })
    
    it('evaluates multiple intervals', async () => {
        const exp = tf.tensor([4.321928094887363, 2.584962500721156])
        const res = harmonicDistance(TRIAD.squeeze())
        await expectTensorsClose(res, exp)
    })    
})

describe('harmonicDistanceAggregate', () => {
    it('evaluates FIFTH', async () => {
        const exp = tf.tensor([2.584962500721156, 2.584962500721156])
        const input = tf.stack([FIFTH, FIFTH], 0)
        const res = await harmonicDistanceAggregate(input)
        await expectTensorsClose(res, exp)
    })

    it('evaluates UNISON_FIFTH', async () => {
        const exp = tf.tensor([2.584962500721156])
        const res = await harmonicDistanceAggregate(UNISON_FIFTH)
        await expectTensorsClose(res, exp)
    })
    
    it('evaluates TRIAD', async () => {
        const exp = tf.tensor([5.9068906])
        const res = await harmonicDistanceAggregate(TRIAD)
        await expectTensorsClose(res, exp)
    })

    it('evalutes two in a row', async () => {
        const exp = tf.tensor([2.584962500721156, 5.9068906])
        const input = tf.concat([UNISON_FIFTH, TRIAD])
        const res = await harmonicDistanceAggregate(input)
        await expectTensorsClose(res, exp)
    })
})
