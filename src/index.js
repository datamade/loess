import { create, all } from 'mathjs'
import sort from 'lodash.sortby'
import gaussian from 'gaussian'
import { validateModel, validatePredict, validateGrid } from './inputsValidation.js'
import {
  weightFunc, normalize, transpose, distMatrix, weightMatrix,
  polynomialExpansion, weightedLeastSquare
} from './helpers.js'
// import data from '../data/gas.json'

const config = { }
const math = create(all, config)

export default class Loess {
  constructor (data, options = {}) {
    Object.assign(this, validateModel(data, options))

    if (this.options.normalize) this.normalization = this.x.map(normalize)

    this.expandedX = polynomialExpansion(this.x, this.options.degree)
    const normalized = this.normalization
      ? this.x.map((x, idx) => this.normalization[idx](x))
      : this.x
    this.transposedX = transpose(normalized)
  }

  predict (data) {
    const { xNew, n } = validatePredict.bind(this)(data)

    const expandedX = polynomialExpansion(xNew, this.options.degree)
    const normalized = this.normalization ? xNew.map((x, idx) => this.normalization[idx](x)) : xNew
    const distM = distMatrix(transpose(normalized), this.transposedX)
    const weightM = weightMatrix(distM, this.w, this.bandwidth)

    let fitted, residuals, weights, betas
    function iterate (wt) {
      fitted = []
      residuals = []
      betas = []
      weights = math.dotMultiply(wt, weightM)
      transpose(expandedX).forEach((point, idx) => {
        const fit = weightedLeastSquare(this.expandedX, this.y, weights[idx])
        if (fit.error) {
          const sumWeights = math.sum(weights[idx])
          const mle = sumWeights === 0 ? 0 : math.multiply(this.y, weights[idx]) / sumWeights
          fit.beta = math.zeros(this.expandedX.length).set([0], mle)
          fit.residual = math.subtract(this.y, mle)
        }
        fitted.push(math.squeeze(math.multiply(point, fit.beta)))
        residuals.push(fit.residual)
        betas.push(fit.beta.toArray())
        const median = math.median(math.abs(fit.residual))
        wt[idx] = fit.residual.map(r => weightFunc(r, 6 * median, 2))
      })
    }

    const robustWeights = Array(n).fill(math.ones(this.n))
    for (let iter = 0; iter < this.options.iterations; iter++) iterate.bind(this)(robustWeights)

    const output = { fitted, betas, weights }

    if (this.options.band) {
      const z = gaussian(0, 1).ppf(1 - (1 - this.options.band) / 2)
      const halfwidth = weights.map((weight, idx) => {
        const V1 = math.sum(weight)
        const V2 = math.multiply(weight, weight)
        const intervalEstimate = Math.sqrt(math.multiply(math.square(residuals[idx]), weight) / (V1 - V2 / V1))
        return intervalEstimate * z
      })
      Object.assign(output, { halfwidth })
    }

    return output
  }

  grid (cuts) {
    validateGrid.bind(this)(cuts)

    const xNew = []
    const xCuts = []
    this.x.forEach((x, idx) => {
      const xSorted = sort(x)
      const xMin = xSorted[0]
      const xMax = xSorted[this.n - 1]
      const width = (xMax - xMin) / (cuts[idx] - 1)
      xCuts.push([])
      for (let i = 0; i < cuts[idx]; i++) xCuts[idx].push(xMin + i * width)

      let repeats = 1
      let copies = 1
      for (let i = idx - 1; i >= 0; i--) repeats *= cuts[i]
      for (let i = idx + 1; i < this.d; i++) copies *= cuts[i]

      xNew.push([])
      for (let i = 0; i < repeats; i++) {
        xNew[idx] = xNew[idx].concat(xCuts[idx].reduce((acc, cut) => acc.concat(Array(copies).fill(cut)), []))
      }
    })

    const data = { x: xNew[0], xCut: xCuts[0] }
    if (this.d > 1) Object.assign(data, { x2: xNew[1], xCut2: xCuts[1] })
    return data
  }
}

// const w = data.NOx.map(() => Math.random() * 10)
// const fit = new Loess({y: data.NOx, x: data.E, w}, {span: 0.8, band: 0.8, degree: 'quadratic'})
// console.log(JSON.stringify(fit.predict(fit.grid([30]))))
