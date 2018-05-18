package com.lucarc.scalaesn.readouts.implementation

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.storage.Zero
import com.lucarc.scalaesn.readouts.Readout
import org.slf4j.{Logger, LoggerFactory}

class LinearRegression(reg: Double, c: Double) extends Readout {
  val _log: Logger = LoggerFactory.getLogger(LinearRegression.super.toString)

  override var weights: DenseMatrix[Double] = _

  override def apply(x: DenseVector[Double]): DenseVector[Double] = {
    val y: DenseVector[Double] = weights * x
    y
  }

  override def train(x: Seq[DenseVector[Double]], yExpected: Seq[DenseVector[Double]]): Unit = {

    _log.info("Training >>>")
    implicit val z: Zero[Double] = breeze.storage.Zero.DoubleZero
    val xMatrix: DenseMatrix[Double] = DenseMatrix(x.flatMap(_.data)).reshape(x.length, x.head.length)
    val yMatrix: DenseMatrix[Double] = DenseMatrix(yExpected.flatMap(_.data)).reshape(yExpected.length, yExpected.head.length)
    val y1: DenseMatrix[Double] = xMatrix.t * xMatrix +  DenseMatrix.eye(xMatrix.cols) *:* reg
    _log.info("Inverting Matrix")
    val y4 = breeze.linalg.inv(y1)
    val y5 = xMatrix.t * yMatrix
    weights = (y4 * y5).t
    _log.info("Training <<<")

  }
}
