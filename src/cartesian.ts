import * as tf from "@tensorflow/tfjs-node-gpu"

const meshGrid = (xi: tf.Tensor[]): tf.Tensor[] => {
    const ndim = xi.length
    const finalShape = xi.map(x => x.shape[0])
    const output = xi.map((x, i) => {
        let newShape = []
        for (let idx = 0; idx < ndim; idx++) {
            if (idx == i) {
                newShape.push(finalShape[i])
            } else {
                newShape.push(1)
            }
        }
        return x.reshape(newShape).broadcastTo(finalShape)
    })
    return output
}

const permutations = (a: tf.Tensor, times = 2): tf.Tensor => {
    if (times == 1) {
        return a.expandDims(1)
    }
    
    const options = tf.range(0, a.shape[0])
    const meshed = meshGrid((new Array(times)).fill(options))
    const indices = tf.stack(meshed, -1)
        .reshape([-1, times])
        .cast("int32")
    return tf.gather(a, indices)
}

export {
    meshGrid,
    permutations,
}
