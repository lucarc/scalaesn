package com.lucarc.scalaesn.readouts.implementation
import breeze.linalg.DenseMatrix

class LR {


  private var computedThetas: DenseMatrix[Double] = _
  implicit val zero

  def fit(x: DenseMatrix[Double], y: DenseMatrix[Double], lr: Float, iters: Int): DenseMatrix[Double] = {
    val bias : DenseMatrix[Double]= DenseMatrix.ones(x.rows, 1)
    val xWithBias: DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.horzcat(DenseMatrix.ones(1,1), bias), x)
    val thetas: DenseMatrix[Double] =  DenseMatrix.zeros(1, xWithBias.cols).reshape(1, xWithBias.cols)
    computedThetas = computeGradient(xWithBias, y, thetas, lr, iters).asInstanceOf[DenseMatrix[Double]]
    computedThetas
  }

  def predict(x: DenseMatrix[Double]): Float = {
    val bias = DenseMatrix.ones(x.rows, 1)
    val xWithBias: DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.horzcat(DenseMatrix.ones(1,1), bias), x)
    xWithBias * computedThetas.t.data(0)
  }

  private def computeCost(x: DenseMatrix[Double], y: DenseMatrix[Double], thetas: DenseMatrix[Double]): Double = {
    val cost: DenseMatrix[Double] = (x * thetas.t - y) *:* (x * thetas.t - y)
    (cost:+= 0 / (2*cost.size)).data(0)
  }

  private def computeGradient(x: DenseMatrix[Double], y: DenseMatrix[Double], thetas: DenseMatrix[Double], lr: Float, iterations: Int): DenseMatrix[Double] ={
    val nbOfTrainingExamples = x.rows
    val computedThetas = (0 to iterations).foldLeft(thetas)({
      case (`thetas`, i) =>
        val error: DenseMatrix[Double] = x * thetas.t - y
        val grad: DenseMatrix[Double] = error.t * x /:/ nbOfTrainingExamples
        val updatedThetas = thetas - (grad *:* lr)
        println(s"Iteration $i cost: ${computeCost(x, y, updatedThetas)}")

        updatedThetas
    })
    computedThetas
  }

}