package com.lucarc.scalaesn.layers

import breeze.linalg.{CSCMatrix, SparseVector}

@SerialVersionUID(101L)
trait Reservoir extends Serializable {

  val reservoir: CSCMatrix[Double]
  val inputLayer: CSCMatrix[Double]
  var v_t_1: SparseVector[Double]

  def activate(x: SparseVector[Double]): SparseVector[Double]
}
