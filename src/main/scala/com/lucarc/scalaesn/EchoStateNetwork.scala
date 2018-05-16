package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.{Rand, RandBasis}

class EchoStateNetwork(nInput: Int, nNeurons: Int, sr: Double, sp: Double, seed: Int = 42) extends ESN  {

  val rand: Rand[Double] = RandBasis.withSeed(seed).uniform

  val Wtemp: DenseMatrix[Double] = DenseMatrix.rand(nNeurons, nNeurons, rand)
  val Wsr: Double  = breeze.linalg.max(breeze.linalg.eig(Wtemp).eigenvalues)
  val W: DenseMatrix[Double] = (Wtemp /:/ Wsr) * sr
  val Win: DenseMatrix[Double] = DenseMatrix.rand(nInput, nNeurons, rand)

  var v_t_1: DenseMatrix[Double] = DenseMatrix.rand(1, nNeurons, rand)



  def activate(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    x * Win
  }

  def activateReservoir(vIn: DenseMatrix[Double], v_1: DenseMatrix[Double]): DenseMatrix[Double] = {
    vIn * W
  }

  override def reservoir: DenseMatrix[Double] = W
  override def inputLayer: DenseMatrix[Double] = Win
}