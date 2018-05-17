package com.lucarc.scalaesn

import breeze.linalg.{DenseMatrix, DenseVector}

trait EchoStateNetwork {
  def fit(x: DenseMatrix[Double], yExpected: DenseMatrix[Double]): Unit
  def transform(x: DenseVector[Double]): DenseVector[Double]
}
