package com.lucarc.scalaesn.layers

import breeze.linalg.DenseMatrix

trait Reservoir {

  def reservoir: DenseMatrix[Double]
  def inputLayer: DenseMatrix[Double]

  def train(x: DenseMatrix[Double])
  def predict(x: DenseMatrix[Double]): DenseMatrix[Double]
}
