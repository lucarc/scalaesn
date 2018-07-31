package com.lucarc.scalaesn.readouts

import breeze.linalg.{CSCMatrix, SparseVector}

trait Readout {
  var weights: CSCMatrix[Double]

  def apply(x: SparseVector[Double]): SparseVector[Double]

  def train(x: Seq[SparseVector[Double]], yExpected: Seq[SparseVector[Double]]): Unit

}
