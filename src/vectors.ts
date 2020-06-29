import * as tf from "@tensorflow/tfjs-node-gpu"

import { meshGrid } from "./cartesian"

const PRIME_LIMITS = [5, 5, 3, 3, 2, 1]
const PRIMES: tf.Tensor = tf.tensor([2, 3, 5, 7, 11, 13, 19, 23])
const PD_BOUNDS = [0.0, 4.0]
const HD_LIMIT = 9.0
const DIMS = 1

const pitchDistance = (vectors: tf.Tensor): tf.Tensor => {
    const primeSlice: tf.Tensor = PRIMES.slice(0, vectors.shape[vectors.shape.length-1])
    const floatRatios = primeSlice
        .asType('float32')
        .pow(vectors)
        .prod(1)
    return floatRatios.log().div(tf.log([2.0]))
}

const restrictBounds = (vectors: tf.Tensor, bounds: number[]): Promise<tf.Tensor> => {
    let pds = pitchDistance(vectors)
    let mask = tf.logicalAnd(
        tf.lessEqual(pds, bounds[1]),
        tf.greaterEqual(pds, bounds[0])
    )
    return tf.booleanMaskAsync(vectors, mask)
}

const spaceGraphAlteredPermutations = async (limits: number[], bounds: number[] = undefined): Promise<tf.Tensor> => {
    const options = limits.map(i => tf.range(-i, i+1))
    let mesh = meshGrid(options)
    let vectors = tf.stack(mesh, -1)
    vectors = vectors.reshape([-1, limits.length]).cast('float32')
    if (bounds) {
        vectors = await restrictBounds(vectors, bounds)
    }
    return vectors
}

class VectorSpace {
    constructor(primeLimits: []) {
    }

    getPerms(primeLimits: number[], bounds: number[] = PD_BOUNDS) {
        const vectors = spaceGraphAlteredPermutations(primeLimits, bounds)
    }
}

export {
    PRIMES,
    pitchDistance,
    restrictBounds,
    spaceGraphAlteredPermutations,
    VectorSpace,
}
