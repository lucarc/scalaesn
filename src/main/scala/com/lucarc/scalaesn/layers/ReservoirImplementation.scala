package com.lucarc.scalaesn.layers

import breeze.linalg.{CSCMatrix, DenseMatrix, DenseVector, SparseVector}
import breeze.stats.distributions.{Rand, RandBasis}
import com.lucarc.scalaesn.activations.ActivationFunction

class ReservoirImplementation(nInput: Int, nNeurons: Int, sr: Double, sp: Double, seed: Int = 42, activation: ActivationFunction) extends Reservoir {

  val rand: Rand[Double] = RandBasis.withSeed(seed).uniform

  val Wtemp: DenseMatrix[Double] = DenseMatrix.rand(nNeurons, nNeurons, rand)

  // make the reservoir sparse
  for (i <- 0 until nNeurons * nNeurons) if (rand.draw() > sp) Wtemp.data.update(i, 0)
  val Wsr: Double = breeze.linalg.max(breeze.linalg.eig(Wtemp).eigenvalues)
  val reservoirD: DenseMatrix[Double] = (Wtemp /:/ Wsr) * sr

  override val reservoir: CSCMatrix[Double] = CSCMatrix.create(reservoirD.rows, reservoirD.cols, reservoirD.data)
  override val inputLayer: CSCMatrix[Double] = CSCMatrix.rand(nNeurons, nInput, rand)

  var v_t_1: SparseVector[Double] = SparseVector(DenseVector.rand(nNeurons, rand).data)



  override def activate(x: SparseVector[Double]): SparseVector[Double] = {
    val v_t_preact: SparseVector[Double] = reservoir * v_t_1 + inputLayer * x
    val v_t: SparseVector[Double] = v_t_preact.map(vi => activation.activate(vi))
    v_t_1 = v_t.copy
    v_t
  }




}