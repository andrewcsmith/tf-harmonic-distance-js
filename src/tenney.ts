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
    const ehd = explodedHarmonicDistance(vecs)
    if (vecs.shape[1] == 1) {
        return ehd.abs().sum(1).sum(1)
    } else {
        const bases = (await getBases(vecs.shape[1] + 1))
            .broadcastTo(vecs.shape)
        return ehd
            .matMul(bases, true, false)
            .abs()
            .sum(1)
            .sum(1)
            .div(vecs.shape[1])
    }
}

export {
    harmonicDistance,
    harmonicDistanceAggregate,
}
