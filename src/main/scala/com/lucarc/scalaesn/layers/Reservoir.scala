package com.lucarc.scalaesn.layers

import breeze.linalg.{DenseMatrix, DenseVector}

trait Reservoir {

  val reservoir: DenseMatrix[Double]
  val inputLayer: DenseMatrix[Double]

  var v_t_1: DenseVector[Double]

  def activate(x: DenseVector[Double]): DenseVector[Double]
}
