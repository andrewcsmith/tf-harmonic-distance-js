import * as tf from "@tensorflow/tfjs-node-gpu"
import { expectArraysEqual } from "@tensorflow/tfjs-core/src/test_util"
import { tensorsAlmostEqual } from "./test_utils"

import { 
    pitchDistance, 
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

describe('spaceGraphAlteredPermutations', () => {
    it('generates permutations for each limit', () => {
        const limits = [1, 2]
        const exp = tf.tensor([
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [ 0, -2], [ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2],
            [ 1, -2], [ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2]
        ])
        const res = spaceGraphAlteredPermutations(limits, [-1.0, 1.0])
        expect(res.arraySync()).toEqual(exp.arraySync())
    })

    it('works with 3 prime dimensions', () => {
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
        const res = spaceGraphAlteredPermutations(limits, [-1.0, 1.0])
        expect(res.arraySync()).toEqual(exp.arraySync())
    })
})
