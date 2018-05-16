package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix

trait ESN {

  def reservoir: DenseMatrix[Double]
  def inputLayer: DenseMatrix[Double]
}
