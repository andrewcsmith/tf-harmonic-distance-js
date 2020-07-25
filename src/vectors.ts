import * as tf from "@tensorflow/tfjs-node-gpu"

import { meshGrid, permutations } from "./cartesian"
import { harmonicDistance, harmonicDistanceAggregate } from "./tenney"

const PRIME_LIMITS = [5, 5, 3, 3, 2, 1]
const PRIMES: tf.Tensor = tf.tensor([2, 3, 5, 7, 11, 13, 19, 23])
const PD_BOUNDS = [0.0, 4.0]
const HD_LIMIT = 9.0
const DIMS = 1

const pitchDistance = (vectors: tf.Tensor): tf.Tensor => {
    return PRIMES.slice(0, vectors.shape.slice(-1))
        .asType('float32')
        .pow(vectors)
        .prod(-1)
        .log()
        .div(tf.log([2.0]))
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

interface VectorSpaceParameters {
    primeLimits ?: number[],
    bounds ?: number[],
    hdLimit ?: number,
    dimensions ?: number,
}

class VectorSpace {
    bounds: number[]
    dimensions: number
    hdLimit: number
    hds: tf.Variable
    pds: tf.Variable
    perms: tf.Variable
    primeLimits: number[]
    twoHds: tf.Tensor
    vectors: tf.Tensor

    constructor({
        primeLimits = PRIME_LIMITS, 
        bounds = PD_BOUNDS, 
        hdLimit = HD_LIMIT, 
        dimensions = DIMS
    }: VectorSpaceParameters) {
        this.primeLimits = primeLimits
        this.bounds = bounds
        this.hdLimit = hdLimit
        this.dimensions = dimensions
    }

    init = async () => {
        this.perms = tf.variable(await this.getPerms(this.primeLimits, this.bounds, this.hdLimit, this.dimensions), true)
        this.hds = tf.variable(await harmonicDistanceAggregate(this.perms))
        this.pds = tf.variable(pitchDistance(this.perms))
        this.twoHds = tf.pow(2.0, this.hds)
    }

    getPerms = async (primeLimits: number[], bounds: number[] = PD_BOUNDS, hdLimit = HD_LIMIT, dimensions = DIMS): Promise<tf.Tensor> => {
        const vectors = await spaceGraphAlteredPermutations(primeLimits, bounds)
        const vectorsHds = harmonicDistance(vectors)
        const mask = tf.lessEqual(vectorsHds, tf.broadcastTo(hdLimit, vectorsHds.shape))
        this.vectors = await tf.booleanMaskAsync(vectors, mask)
        return permutations(this.vectors, dimensions)
    }
}

export {
    PRIMES,
    pitchDistance,
    restrictBounds,
    spaceGraphAlteredPermutations,
    VectorSpace,
}
