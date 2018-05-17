package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix

trait EchoStateNetwork {
  var v_t_1: DenseMatrix[Double]
  def fit(x: DenseMatrix[Double], yExpected: DenseMatrix[Double]): Unit
  def transform(x: DenseMatrix[Double]): DenseMatrix[Double]
}
