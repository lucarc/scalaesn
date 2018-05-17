package com.lucarc.scalaesn.readouts.implementation

import breeze.linalg.DenseMatrix
import breeze.storage.Zero
import com.lucarc.scalaesn.readouts.Readout

class LinearRegression(reg: Double, c: Double) extends Readout {
  override var weights: DenseMatrix[Double] = _

  override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val y: DenseMatrix[Double] = x * weights
    y
  }

  override def train(x: DenseMatrix[Double], yExpected: DenseMatrix[Double]): Unit = {
    val y1: DenseMatrix[Double] = x.t * x
    implicit val zero: Zero[Double] = breeze.storage.Zero.DoubleZero
    val y2: DenseMatrix[Double] = DenseMatrix.eye(x.cols) *:* reg
    val y3 = y1 + y2
    val y4 = breeze.linalg.pinv(y3)
    val y5 = x.t * yExpected
    weights = y4 * y5
  }
}
