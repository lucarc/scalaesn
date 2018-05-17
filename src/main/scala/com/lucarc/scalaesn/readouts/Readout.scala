package com.lucarc.scalaesn.readouts

import breeze.linalg.DenseMatrix

trait Readout {
  var weights: DenseMatrix[Double]
  def apply(x: DenseMatrix[Double]): DenseMatrix[Double]
  def train(x: DenseMatrix[Double], yExpected:  DenseMatrix[Double]): Unit

}
