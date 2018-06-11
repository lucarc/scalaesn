package com.lucarc.scalaesn
import breeze.linalg.SparseVector
import com.lucarc.scalaesn.layers.Reservoir
import com.lucarc.scalaesn.readouts.Readout
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.parallel.mutable.ParArray

class EchoStateNetworkImplementation(reservoir: Reservoir, readout: Readout) extends EchoStateNetwork {
  val _log: Logger = LoggerFactory.getLogger(EchoStateNetworkImplementation.super.toString)


  override def fit(x: Seq[SparseVector[Double]], yExpected: Seq[SparseVector[Double]], numThreads: Int): Unit = {
    _log.info("Fitting >>>")
    val parX: ParArray[SparseVector[Double]] = ParArray.fromTraversables(x)
    parX.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(numThreads))
    val start: Long = System.currentTimeMillis()
    _log.info("Computing V...")
    val v = parX.map(xSample => reservoir.activate(xSample))
    _log.info(s"Reservoir activations computed in ${System.currentTimeMillis() - start} milliseconds")
    _log.info("Training Readout...")
    readout.train(v.seq, yExpected)
    _log.info("Fitting <<<")

  }

  override def transform(x: SparseVector[Double]): SparseVector[Double] = {
    readout.apply(reservoir.activate(x))
  }
}