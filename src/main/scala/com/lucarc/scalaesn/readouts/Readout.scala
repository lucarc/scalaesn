package com.lucarc.scalaesn.readouts

import breeze.linalg.{CSCMatrix, SparseVector}

@SerialVersionUID(102L)
trait Readout extends Serializable {
  var weights: CSCMatrix[Double]
  def apply(x: SparseVector[Double]): SparseVector[Double]
  def train(x: Seq[SparseVector[Double]], yExpected:  Seq[SparseVector[Double]]): Unit

}
