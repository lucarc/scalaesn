package com.lucarc.scalaesn.readouts.implementation
import breeze.linalg.DenseMatrix

class LR (reg: Double) {

  var weights: DenseMatrix[Double]

  def train(input: DenseMatrix[Double], output: DenseMatrix[Double]): Unit = {
    val y1: DenseMatrix[Double] = input.t * input
    val y2: DenseMatrix[Double] = DenseMatrix.eye(input.cols) *:* reg
    val y3 = y1 + y2
    val y4 = breeze.linalg.pinv(y3)
    val y5 = input.t * output
    weights = y4 * y5
  }


}