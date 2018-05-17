package com.lucarc.scalaesn.readouts

import breeze.linalg.DenseMatrix

trait Readout {
  var weights: DenseMatrix[Double]
  def apply(x: DenseMatrix[Double]): Any
  def train(x: DenseMatrix[Double], yExpected:  DenseMatrix[Double]): Unit

}
