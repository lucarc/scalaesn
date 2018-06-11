package com.lucarc.scalaesn

import breeze.linalg.SparseVector

import scala.collection.parallel.ParSeq

trait EchoStateNetwork {
  def fit(x: Seq[SparseVector[Double]], yExpected: Seq[SparseVector[Double]], numThreads: Int): Unit
  def fit(x: ParSeq[SparseVector[Double]], yExpected: ParSeq[SparseVector[Double]], numThreads: Int): Unit
  def transform(x: SparseVector[Double]): SparseVector[Double]
}
