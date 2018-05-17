package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix
import com.lucarc.scalaesn.layers.ReservoirImplementation
import com.lucarc.scalaesn.readouts.Readout

class EchoStateNetworkPredictor(echoStateNetwork: ReservoirImplementation, readout: Readout) extends EchoStateNetwork {


  override def fit(x: DenseMatrix[Double], yExpected: DenseMatrix[Double]): Unit = {
    echoStateNetwork.activate(x)
  }

  override def transform(x: DenseMatrix[Double]): DenseMatrix[Double] = ???
}