package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix
import com.lucarc.scalaesn.layers.Reservoir
import com.lucarc.scalaesn.readouts.Readout

class EchoStateNetworkPredictor(reservoir: Reservoir, readout: Readout) extends EchoStateNetwork {

  override var v_t_1: DenseMatrix[Double] = _

  override def fit(x: DenseMatrix[Double], yExpected: DenseMatrix[Double]): Unit = {
    val v = reservoir.activate(x)
    readout.train(v, yExpected)
  }

  override def transform(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    readout.apply(reservoir.activate(x))
  }
}