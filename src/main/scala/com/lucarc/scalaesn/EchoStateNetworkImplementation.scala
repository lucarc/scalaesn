package com.lucarc.scalaesn

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.lucarc.scalaesn.layers.Reservoir
import com.lucarc.scalaesn.readouts.Readout

class EchoStateNetworkImplementation(reservoir: Reservoir, readout: Readout) extends EchoStateNetwork {


  override def fit(x: Seq[DenseVector[Double]], yExpected: Seq[DenseVector[Double]]): Unit = {
    val v = x.map(xSample => reservoir.activate(xSample))
    readout.train(v, yExpected)
  }

  override def transform(x: DenseVector[Double]): DenseVector[Double] = {
    readout.apply(reservoir.activate(x))
  }
}