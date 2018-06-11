package com.lucarc.scalaesn.readouts.implementation

import breeze.linalg.{CSCMatrix, DenseMatrix, SparseVector}
import breeze.storage.Zero
import com.lucarc.scalaesn.readouts.Readout
import org.slf4j.{Logger, LoggerFactory}

class LinearRegression(reg: Double, c: Double) extends Readout {
  val _log: Logger = LoggerFactory.getLogger(LinearRegression.super.toString)

  override var weights: CSCMatrix[Double] = _

  override def apply(x: SparseVector[Double]): SparseVector[Double] = {
    val y: SparseVector[Double] = weights * x
    y
  }

  override def train(x: Seq[SparseVector[Double]], yExpected: Seq[SparseVector[Double]]): Unit = {

    _log.info("Training >>>")
    val start: Long = System.currentTimeMillis()
    implicit val z: Zero[Double] = breeze.storage.Zero.DoubleZero
    _log.info("Creating xMatrix")
    _log.info("Creating yMatrix")
    val xMatrix: CSCMatrix[Double] = CSCMatrix.create[Double](x.length, x.head.length, x.flatMap(_.data).toArray)
    val yMatrix: CSCMatrix[Double] = CSCMatrix.create[Double](yExpected.length, yExpected.head.length, yExpected.flatMap(_.data).toArray)

    _log.info("yMatrix y1")
    val eye: DenseMatrix[Double] = DenseMatrix.eye(xMatrix.cols)
    val eyeCSC: CSCMatrix[Double] = CSCMatrix.create(eye.rows, eye.cols, eye.data)
    val y1: CSCMatrix[Double] = xMatrix.t * xMatrix + eyeCSC *:* reg

    _log.info("Inverting Matrix and computing weights")
    weights = (y1 \ (xMatrix.t * yMatrix)).t
    _log.info(s"Readout Trained in ${System.currentTimeMillis() - start} milliseconds")


    _log.info("Training <<<")
  }
}
