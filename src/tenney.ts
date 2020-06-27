import * as tf from "@tensorflow/tfjs-node-gpu"
import { PRIMES } from "./vectors"
import { getBases } from "./utilities"

const explodedHarmonicDistance = (vecs: tf.Tensor): tf.Tensor => {
    let out = PRIMES.slice(0, vecs.shape.slice(-1))
    out = tf.pow(out, vecs)
    return out.log().div(tf.log(2.0))
}

const harmonicDistance = (vecs: tf.Tensor): tf.Tensor => {
    return tf.sum(tf.abs(explodedHarmonicDistance(vecs)), 1)
}

const harmonicDistanceAggregate = async (vecs: tf.Tensor): Promise<tf.Tensor> => {
    const bases = await getBases(vecs.shape[0] + 1)
    return explodedHarmonicDistance(vecs)
        .transpose()
        .dot(bases)
        .abs()
        .sum(0)
        .sum(0)
        .div(vecs.shape[0])
}

export {
    harmonicDistance,
    harmonicDistanceAggregate,
}
