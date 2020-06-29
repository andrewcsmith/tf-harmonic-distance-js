import * as tf from "@tensorflow/tfjs-node-gpu"
import { expectArraysEqual } from "@tensorflow/tfjs-core/src/test_util"
import { tensorsAlmostEqual } from "./test_utils"

import { 
    pitchDistance, 
    restrictBounds,
    spaceGraphAlteredPermutations,
} from "../src/vectors"

describe('pitchDistance', () => {
    it('is correct for one vec', () => {
        const v = tf.tensor([[-1, 1, 0]])
        const exp = 0.58
        const res = pitchDistance(v)
        expect(res.arraySync()[0]).toBeCloseTo(exp)
    })
    
    test('is correct for multiple vecs', () => {
        const v = tf.tensor([[-1, 1, 0], [-2, 0, 1]])
        const exp = tf.tensor([0.5849625, 0.3219281])
        const res = pitchDistance(v)
        expect(tensorsAlmostEqual(res, exp)).toBe(true)
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
        expect(tensorsAlmostEqual(res, exp)).toBe(true)
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
        expect(tensorsAlmostEqual(res, exp)).toBe(true)
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
        expect(tensorsAlmostEqual(res, exp)).toBe(true)
    })
})
