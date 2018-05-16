package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix

trait ESN {

  def reservoir: DenseMatrix[Double]
  def inputLayer: DenseMatrix[Double]

  def train(x: DenseMatrix[Double])
  def predict(x: DenseMatrix[Double]): DenseMatrix[Double]
}
