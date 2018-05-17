package com.lucarc.scalaesn.readouts.implementation

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import com.lucarc.scalaesn.readouts.Readout

class LinearRegression(reg: Double, c: Double) extends Readout {
  override var weights: DenseMatrix[Double] = _

  override def apply(x: DenseVector[Double]): DenseVector[Double] = {
    val y: DenseVector[Double] = weights * x
    y
  }

  override def train(x: Seq[DenseVector[Double]], yExpected: Seq[DenseVector[Double]]): Unit = {

    implicit val z: Zero[Double] = breeze.storage.Zero.DoubleZero
    val xMatrix: DenseMatrix[Double] = DenseMatrix(x.flatMap(_.data)).reshape(x.length, x.head.length)
    val yMatrix: DenseMatrix[Double] = DenseMatrix(yExpected.flatMap(_.data)).reshape(yExpected.length, yExpected.head.length)
    val y1: DenseMatrix[Double] = xMatrix.t * xMatrix

    val y2: DenseMatrix[Double] = DenseMatrix.eye(xMatrix.cols) *:* reg
    val y3 = y1 + y2
    val y4 = breeze.linalg.pinv(y3)
    val y5 = xMatrix.t * yMatrix
    weights = y4 * y5
  }
}
