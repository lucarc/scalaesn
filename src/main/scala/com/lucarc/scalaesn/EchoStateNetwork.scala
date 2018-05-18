package com.lucarc.scalaesn

import breeze.linalg.{DenseVector}

trait EchoStateNetwork {
  def fit(x: Seq[DenseVector[Double]], yExpected: Seq[DenseVector[Double]]): Unit
  def transform(x: DenseVector[Double]): DenseVector[Double]
}
