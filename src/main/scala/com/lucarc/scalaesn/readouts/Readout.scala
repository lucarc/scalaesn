package com.lucarc.scalaesn.readouts

import breeze.linalg.{CSCMatrix, DenseVector}

trait Readout {
  var weights: CSCMatrix[Double]
  def apply(x: DenseVector[Double]): DenseVector[Double]
  def train(x: Seq[DenseVector[Double]], yExpected:  Seq[DenseVector[Double]]): Unit

}
