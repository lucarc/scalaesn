package com.lucarc.scalaesn

import breeze.linalg.SparseVector

trait EchoStateNetwork {
  def fit(x: Seq[SparseVector[Double]], yExpected: Seq[SparseVector[Double]]): Unit
  def transform(x: SparseVector[Double]): SparseVector[Double]
}
