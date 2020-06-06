function tensorsAlmostEqual(res, exp, range = 1.0e-5) {
    return res.sub(exp).abs().lessEqual(range).all().dataSync()[0] == 1
}

export {
    tensorsAlmostEqual,
}
