package com.lucarc.scalaesn

import breeze.linalg.SparseVector
import com.lucarc.scalaesn.activations.Sigmoid
import com.lucarc.scalaesn.layers.ReservoirImplementation
import com.lucarc.scalaesn.readouts.implementation.LinearRegression

object Main {

  val numTrain: Int = 50000
  val numTest: Int = 1000

  val t: Int = 3

  val trainX: Seq[SparseVector[Double]] = (0 until numTrain).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl(deg)))
  val trainY: Seq[SparseVector[Double]] = (t until numTrain + t).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl(deg)))

  val testX: Seq[SparseVector[Double]] = (0 until numTest).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl(deg)))
  val testY: Seq[SparseVector[Double]] = (t until numTest + t).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl(deg)))

  val spectralRadius: Double = 0.7
  val sparsity: Double = 0.80

  val washout: Int = 15


  def main(args: Array[String]): Unit = {

    val esn: EchoStateNetwork = new EchoStateNetworkImplementation(reservoir = new ReservoirImplementation(nInput = 1, nNeurons = 100, sr = spectralRadius, sp = sparsity, activation = new Sigmoid), readout = new LinearRegression(reg = 1, c = 1))
    esn.fit(trainX, trainY)

    val predictedTestY: Seq[SparseVector[Double]] = testX.map(xi => esn.transform(x = xi))
    val unrolledData = (testY zip predictedTestY).flatMap(yVEP => yVEP._1.data zip yVEP._2.data)
    val washedOutData = unrolledData.drop(washout)

    val mse: Double = washedOutData.map(y => (y._1 - y._2) * (y._1 - y._2)).sum / washedOutData.length
    System.out.print(mse)
    System.exit(0)
  }
}
