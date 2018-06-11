package com.lucarc.scalaesn

import breeze.linalg.{*, DenseMatrix, DenseVector, SparseVector}
import com.lucarc.scalaesn.layers.Reservoir
import com.lucarc.scalaesn.readouts.Readout
import org.slf4j.{Logger, LoggerFactory}

class EchoStateNetworkImplementation(reservoir: Reservoir, readout: Readout) extends EchoStateNetwork {
  val _log: Logger = LoggerFactory.getLogger(EchoStateNetworkImplementation.super.toString)


  override def fit(x: Seq[SparseVector[Double]], yExpected: Seq[SparseVector[Double]]): Unit = {
    _log.info("Fitting >>>")
    _log.info("Computing V...")
    val v = x.map(xSample => reservoir.activate(xSample))
    _log.info("Training Readout...")
    readout.train(v, yExpected)
    _log.info("Fitting <<<")

  }

  override def transform(x: SparseVector[Double]): SparseVector[Double] = {
    _log.info(s"Applying readout transformation to $x...")
    readout.apply(reservoir.activate(x))
  }
}