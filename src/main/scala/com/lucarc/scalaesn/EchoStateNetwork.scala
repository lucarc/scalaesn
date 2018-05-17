package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix

trait EchoStateNetwork {
  def fit(x: DenseMatrix[Double], yExpected: DenseMatrix[Double]): Unit
  def transform(x: DenseMatrix[Double]): DenseMatrix[Double]
}
