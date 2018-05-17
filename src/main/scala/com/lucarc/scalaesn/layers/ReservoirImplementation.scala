package com.lucarc.scalaesn.layers

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{Rand, RandBasis}

class ReservoirImplementation(nInput: Int, nNeurons: Int, sr: Double, sp: Double, seed: Int = 42) extends Reservoir {

  val rand: Rand[Double] = RandBasis.withSeed(seed).uniform

  val Wtemp: DenseMatrix[Double] = DenseMatrix.rand(nNeurons, nNeurons, rand)

  // make the reservoir sparse
  for (i <- 0 until nNeurons * nNeurons) if (rand.draw() > sp) Wtemp.data.update(i, 0)
  val Wsr: Double = breeze.linalg.max(breeze.linalg.eig(Wtemp).eigenvalues)
  override val reservoir: DenseMatrix[Double] = (Wtemp /:/ Wsr) * sr
  override val inputLayer: DenseMatrix[Double] = DenseMatrix.rand(nInput, nNeurons, rand)

  var v_t_1: DenseVector[Double] = DenseVector.zeros(nNeurons)



  override def activate(x: DenseVector[Double]): DenseVector[Double] = {
    val v_t: DenseVector[Double] = (inputLayer * x) + (reservoir * v_t_1)
    v_t_1 = v_t
    v_t
  }




}