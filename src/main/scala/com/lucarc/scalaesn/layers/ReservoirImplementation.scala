package com.lucarc.scalaesn.layers

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.{Rand, RandBasis}

class ReservoirImplementation(nInput: Int, nNeurons: Int, sr: Double, sp: Double, seed: Int = 42) extends Reservoir {

  val rand: Rand[Double] = RandBasis.withSeed(seed).uniform

  val Wtemp: DenseMatrix[Double] = DenseMatrix.rand(nNeurons, nNeurons, rand)
  val Wsr: Double = breeze.linalg.max(breeze.linalg.eig(Wtemp).eigenvalues)
  override val reservoir: DenseMatrix[Double] = (Wtemp /:/ Wsr) * sr

  // make the reservoir sparse
  for (i <- 0 until nNeurons * nNeurons) if (rand.draw() > sp) reservoir.data.update(i, 0)
  override val inputLayer: DenseMatrix[Double] = DenseMatrix.rand(nInput, nNeurons, rand)

  var v_t_1: DenseMatrix[Double] = DenseMatrix.rand(1, nNeurons, rand)



  override def activate(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val v_t: DenseMatrix[Double] = (x * inputLayer) + (v_t_1 * reservoir)
    v_t_1 = v_t
    v_t
  }




}