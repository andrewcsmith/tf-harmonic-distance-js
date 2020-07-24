import * as tf from "@tensorflow/tfjs-node-gpu"
import { expectTensorsClose } from "./test_utils"

import { 
    pitchDistance, 
    restrictBounds,
    spaceGraphAlteredPermutations,
    VectorSpace,
} from "../src/vectors"

describe('pitchDistance', () => {
    it('is correct for one vec', () => {
        const v = tf.tensor([[-1, 1, 0]])
        const exp = 0.58
        const res = pitchDistance(v)
        expect(res.arraySync()[0]).toBeCloseTo(exp)
    })
    
    test('is correct for multiple vecs', async () => {
        const v = tf.tensor([[-1, 1, 0], [-2, 0, 1]])
        const exp = tf.tensor([0.5849625, 0.3219281])
        const res = pitchDistance(v)
        await expectTensorsClose(res, exp)
    })
})

describe('restrictBounds', () => {
    it('eliminates vectors that are out of bounds', async () => {
        let input = tf.tensor([
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [ 0, -2], [ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2],
            [ 1, -2], [ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2]
        ])
        let bounds = [-1.0, 1.0]
        let exp = tf.tensor([
            [-1, 0], [-1, 1],
            [ 0, 0],
            [1, -1], [ 1, 0],
        ])
        const res = await restrictBounds(input, bounds)
        await expectTensorsClose(res, exp)
    })
})

describe('spaceGraphAlteredPermutations', () => {
    it('generates permutations for each limit', async () => {
        const limits = [1, 2]
        const exp = tf.tensor([
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [ 0, -2], [ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2],
            [ 1, -2], [ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2]
        ])
        const res = await spaceGraphAlteredPermutations(limits)
        await expectTensorsClose(res, exp, false)
    })

    it('works with 3 prime dimensions', async () => {
        const limits = [1, 2, 1]
        const exp = tf.tensor([
            [-1, -2, -1], [-1, -2,  0], [-1, -2,  1], 
            [-1, -1, -1], [-1, -1,  0], [-1, -1,  1], 
            [-1,  0, -1], [-1,  0,  0], [-1,  0,  1], 
            [-1,  1, -1], [-1,  1,  0], [-1,  1,  1], 
            [-1,  2, -1], [-1,  2,  0], [-1,  2,  1], 

            [ 0, -2, -1], [ 0, -2,  0], [ 0, -2,  1], 
            [ 0, -1, -1], [ 0, -1,  0], [ 0, -1,  1], 
            [ 0,  0, -1], [ 0,  0,  0], [ 0,  0,  1], 
            [ 0,  1, -1], [ 0,  1,  0], [ 0,  1,  1], 
            [ 0,  2, -1], [ 0,  2,  0], [ 0,  2,  1], 

            [ 1, -2, -1], [ 1, -2,  0], [ 1, -2,  1], 
            [ 1, -1, -1], [ 1, -1,  0], [ 1, -1,  1], 
            [ 1,  0, -1], [ 1,  0,  0], [ 1,  0,  1], 
            [ 1,  1, -1], [ 1,  1,  0], [ 1,  1,  1], 
            [ 1,  2, -1], [ 1,  2,  0], [ 1,  2,  1]
        ])
        const res = await spaceGraphAlteredPermutations(limits)
        await expectTensorsClose(res, exp, false)
    })
})

describe('VectorSpace', () => {
    it('generates permutations', async () => {
        const exp = tf.tensor([
            [-1, 0], [-1, 1],
            [ 0, 0],
            [1, -1], [ 1, 0],
        ])
        let vs = new VectorSpace([1, 2], [-1.0, 1.0])
        await vs.init()
        await expectTensorsClose(vs.vectors, exp, false)
    })
})
