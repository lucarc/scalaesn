package com.lucarc.scalaesn.readouts

import breeze.linalg.{DenseMatrix, DenseVector}

trait Readout {
  var weights: DenseMatrix[Double]
  def apply(x: DenseVector[Double]): DenseVector[Double]
  def train(x: DenseMatrix[Double], yExpected:  DenseMatrix[Double]): Unit

}
