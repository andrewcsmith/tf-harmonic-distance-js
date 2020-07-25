import * as tf from "@tensorflow/tfjs-node-gpu"

const parabolicLossFunction = (pds: tf.Tensor, hds: tf.Tensor, logPitches: tf.Tensor, curves: tf.Tensor = undefined) => {
    const twoHds = tf.pow(2.0, hds).expandDims(-1)
    const diffs = pds.expandDims(1).sub(logPitches)
    if (curves == undefined) {
        curves = tf.onesLike(diffs)
    }
    
    const reduced = tf.square(diffs)
        .div(curves)
        .sum(-1)
    let scaled = twoHds.mul(reduced)
    scaled = scaled.add(hds.expandDims(-1))
    scaled = scaled.min(0)
    return scaled
}

export {
    parabolicLossFunction
}
