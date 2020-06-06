import * as tf from "@tensorflow/tfjs-node-gpu"

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

export {
    pitchDistance
}
