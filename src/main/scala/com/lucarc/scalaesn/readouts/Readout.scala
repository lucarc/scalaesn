package com.lucarc.scalaesn.readouts

import breeze.linalg.DenseMatrix

trait Readout {

  def apply(x: DenseMatrix[Double]): Any
  def train(x: DenseMatrix[Double])

}
