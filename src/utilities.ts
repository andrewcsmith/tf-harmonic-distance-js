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

const transformToUnitCircle = async (xys) => {
    let [xs, ys] = tf.unstack(xys, 1)
    let theta = tf.atan(ys.div(xs));
    theta = await tf.where(theta.isFinite(), theta, tf.zerosLike(theta))
    let r = tf.sum(tf.square(xys), 1).sqrt()
    let polar_xs = xs.mul(theta.cos())
    let polar_ys = ys.mul(theta.sin())
    let polar_xys = tf.stack([polar_xs, polar_ys], 1)
    r = r.mul(tf.max(polar_xys, 1)).sqrt()
    let new_x = theta.cos().mul(r)
    let new_y = theta.sin().mul(r)
    return tf.stack([new_x, new_y], 1)
}

export { 
    getBases,
    transformToUnitCircle,
}
