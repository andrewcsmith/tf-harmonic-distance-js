import * as tf from "@tensorflow/tfjs-node-gpu"

const selfMesh = (a: tf.Tensor): tf.Tensor[] => {
    const l = a.shape[0]
    const xx = tf.matMul(tf.ones([l, 1]), a.reshape([1, l]))
    const yy = tf.matMul(a.reshape([l, 1]), tf.ones([1, l]))
    return [xx, yy]
}

const permutations = (a: tf.Tensor, times = 2): tf.Tensor => {
    const options = tf.range(0, a.shape[0])
    const meshed = selfMesh(options)
    const indices = tf.stack(meshed, -1).reshape([-1, times]).cast("int32")
    return tf.gather(a, indices)
}

export { 
    permutations,
}
