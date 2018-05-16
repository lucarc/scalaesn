package com.lucarc.scalaesn

import breeze.linalg.DenseMatrix
import com.lucarc.scalaesn.readouts.Readout
import nak.regress.LinearRegression
class EchoStateNetworkPredictor(echoStateNetwork: EchoStateNetwork, readout: Readout) {


  def train(x: DenseMatrix[Double]): Unit = {
    echoStateNetwork.activate(x)
  }

  def predict(x: DenseMatrix[Double]): DenseMatrix[Double] = ???
}