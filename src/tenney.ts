import * as tf from "@tensorflow/tfjs-node-gpu"
import { PRIMES } from "./vectors"

const harmonicDistance = (vecs: tf.Tensor): tf.Tensor => {
    let out = PRIMES.slice(0, vecs.shape.slice(-1))
    out = tf.pow(out, vecs)
    out = out.log().div(tf.log(2.0))
    out = tf.sum(tf.abs(out), 1)
    return out
}

export {
    harmonicDistance
}
