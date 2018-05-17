package com.lucarc.scalaesn.layers

import breeze.linalg.DenseMatrix

trait Reservoir {

  val reservoir: DenseMatrix[Double]
  val inputLayer: DenseMatrix[Double]

  def activate(x: DenseMatrix[Double]): DenseMatrix[Double]
}
