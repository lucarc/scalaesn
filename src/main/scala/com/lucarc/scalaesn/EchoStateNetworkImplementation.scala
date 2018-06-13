package com.lucarc.scalaesn

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import breeze.linalg.SparseVector
import com.lucarc.scalaesn.layers.Reservoir
import com.lucarc.scalaesn.readouts.Readout
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.parallel.mutable.ParArray
import scala.collection.parallel.{ForkJoinTaskSupport, ParSeq}

class EchoStateNetworkImplementation(reservoir: Reservoir, readout: Readout) extends EchoStateNetwork {
  val _log: Logger = LoggerFactory.getLogger(EchoStateNetworkImplementation.super.toString)

  /**
    *
    * @param x          sequence of input vectors
    * @param yExpected  sequence of expected output vetors
    * @param numThreads number of threads on which to parallelise the activation task
    */
  override def fit(x: ParSeq[SparseVector[Double]], yExpected: ParSeq[SparseVector[Double]], numThreads: Int): Unit = {
    _log.info("Fitting >>>")
    x.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(numThreads))
    val start: Long = System.currentTimeMillis()
    _log.info("Computing V...")
    val v = x.map(xSample => reservoir.activate(xSample))
    _log.info(s"Reservoir activations computed in ${System.currentTimeMillis() - start} milliseconds")
    _log.info("Training Readout...")
    readout.train(v.seq, yExpected.seq)
    _log.info("Fitting <<<")

  }

  /**
    *
    * @param x          sequence of input vectors
    * @param yExpected  sequence of expected output vetors
    * @param numThreads number of threads on which to parallelise the activation task
    */
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

  /**
    *
    * @param x input vector
    * @return ouput of the network
    */
  override def transform(x: SparseVector[Double]): SparseVector[Double] = {
    readout.apply(reservoir.activate(x))
  }

  override def dump(filepath: String): Unit = {
    // (2) write the instance out to a file
    val oos = new ObjectOutputStream(new FileOutputStream(filepath))
    oos.writeObject(this)
    oos.close()
  }

  override def load(filepath: String): EchoStateNetwork = {
    // (3) read the object back in
    val ois = new ObjectInputStream(new FileInputStream(filepath))
    val esn = ois.readObject.asInstanceOf[EchoStateNetwork]
    ois.close()
    esn
  }
}