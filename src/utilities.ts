import * as tf from "@tensorflow/tfjs-node-gpu"
import combinations from "combinations"

const combinatorialContour = (vec: Array<number>) => {
    let combos: Array<number> = combinations(vec, 2, 2)
    return combos.map(c => c[0] - c[1])
}

const getBases = async (length: number): Promise<tf.Tensor> => {
    const eye = await tf.eye(length).array()
    let combos = tf.tensor(eye.map(combinatorialContour))
    return combos.slice([1, 0]).mul(-1)
}

export { 
    getBases
}
