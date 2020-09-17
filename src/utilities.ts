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
    if (!ys) { return xys }
    let theta = tf.atan(ys.div(xs));
    theta = await tf.where(theta.isFinite(), theta, tf.zerosLike(theta))
    let r = tf.sum(tf.square(xys), 1).sqrt()
    let polar_xs = xs.mul(theta.cos())
    let polar_ys = ys.mul(theta.sin())
    let polar_xys = tf.max(tf.stack([polar_xs, polar_ys], 1), 1)

    // Scale the r so that [1.0, 1.0] lies on the unit circle
    r = r.mul(polar_xys).sqrt()
    let new_x = theta.cos().mul(r)
    let new_y = theta.sin().mul(r)
    return tf.stack([new_x, new_y], 1)
}

const inverseTransformToUnitCircle = async (xys) => {
    let [xs, ys] = tf.unstack(xys, 1)
    if (!ys) { return xys }
    let theta = tf.atan(ys.div(xs));
    theta = await tf.where(theta.isFinite(), theta, tf.zerosLike(theta))
    let polar_xs = xs.div(theta.cos())
    let polar_ys = ys.div(theta.sin())
    let polar_xys = tf.stack([polar_xs, polar_ys], 1)
    return await tf.where(polar_xys.isFinite(), polar_xys, tf.zerosLike(polar_xys))
}

export { 
    getBases,
    transformToUnitCircle,
    inverseTransformToUnitCircle,
}
