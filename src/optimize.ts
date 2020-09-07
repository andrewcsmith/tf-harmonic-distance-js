import * as tf from "@tensorflow/tfjs-node-gpu"

import { HD_LIMIT, PRIME_LIMITS, PD_BOUNDS, VectorSpace } from "./vectors"

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

interface MinimizerParameters {
    callback?: () => Promise<void>,
    dimensions?: number,
    learningRate?: number,
    maxIters?: number,
    convergenceThreshold ?: number,
    c?: number,
    batchSize?: number,
    bounds?: number[],
    primeLimits?: number[],
    hdLimit?: number,
}

class Minimizer {
    callback: () => Promise<void>
    convergenceThreshold: tf.Scalar
    curves: tf.Tensor
    logPitches: tf.Variable
    maxIters: tf.Scalar
    opt: tf.Optimizer
    step: tf.Variable
    vs: VectorSpace

    constructor({
        callback = async () => {},
        dimensions = 1,
        learningRate = 1e-2,
        maxIters = 1e3,
        convergenceThreshold = 1e-5,
        c = 1e-2,
        batchSize = 1,
        bounds = PD_BOUNDS,
        primeLimits = PRIME_LIMITS,
        hdLimit = HD_LIMIT,
    }: MinimizerParameters) {
        this.callback = callback
        this.convergenceThreshold = tf.scalar(convergenceThreshold)
        this.curves = tf.variable(tf.fill([dimensions], c), false)
        this.logPitches = tf.variable(tf.zeros([batchSize, dimensions], "float32"), true, `logPitches-${batchSize}x${dimensions}`)
        this.maxIters = tf.scalar(maxIters)
        this.opt = tf.train.adam(learningRate)
        this.step = tf.variable(tf.scalar(0), false, undefined, "int32")
        this.vs = new VectorSpace({
            primeLimits,
            hdLimit,
            dimensions,
        })
    }

    minimize = async () => {
        this.step.assign(tf.scalar(0, "int32"))
        await this.reinitializeWeights()
        this.opt.minimize(this.loss, false, [this.logPitches])
        while (await this.stoppingOp().logicalAnd(this.step.less(this.maxIters)).array()) {
            await this.takeStep()
        }
    }

    takeStep = async () => {
        this.opt.minimize(this.loss, false, [this.logPitches])
        this.step.assign(this.step.add(1).toInt())
        await this.callback()
    }

    loss = (): tf.Scalar => {
        const loss = parabolicLossFunction(this.vs.pds, this.vs.hds, this.logPitches, this.curves)
        return loss.sum(0)
    }

    reinitializeWeights = async () => {
        let weights = await this.opt.getWeights()
        for (let weight of weights.slice(1)) {
            weight.tensor = tf.zerosLike(weight.tensor)
        }
        await this.opt.setWeights(weights)
    }

    stoppingOp = () => {
        let { grads } = this.opt.computeGradients(this.loss, [this.logPitches])
        let norm = tf.variable(tf.zeros([this.vs.dimensions]))
        for (const grad of Object.values(grads)) {
            norm.assign(norm.add(tf.pow(grad, 2.0).sum(0)))
        }
        return norm.greaterEqual(this.convergenceThreshold).all().asScalar()
    }
}

export {
    parabolicLossFunction,
    Minimizer,
}
