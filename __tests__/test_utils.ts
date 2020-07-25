import * as tf from "@tensorflow/tfjs-node-gpu"

const expectTensorsClose = async (res: tf.Tensor, exp: tf.Tensor, close = true, epsilon = 0.001) => {
    tf.test_util.expectArraysEqual(res.shape, exp.shape)
    if (close) {
        tf.test_util.expectArraysClose(await res.array(), await exp.array(), epsilon)
    } else {
        tf.test_util.expectArraysEqual(await res.array(), await exp.array())
    }
}

export {
    expectTensorsClose,
}
