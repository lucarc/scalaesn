package com.lucarc.scalaesn

import breeze.linalg.DenseVector
import com.lucarc.scalaesn.activations.Sigmoid
import com.lucarc.scalaesn.layers.ReservoirImplementation
import com.lucarc.scalaesn.readouts.implementation.LinearRegression

object Main {

  val numTrain: Int = 500
  val numTest: Int = 100

  val t: Int = 10

  val trainX: Seq[DenseVector[Double]] = (0 until numTrain).toList.map(deg => DenseVector(breeze.numerics.sin.sinDoubleImpl(deg)))
  val trainY: Seq[DenseVector[Double]] = (t until numTrain + t).toList.map(deg => DenseVector(breeze.numerics.sin.sinDoubleImpl(deg)))

  val testX: Seq[DenseVector[Double]] = (0 until numTest).toList.map(deg => DenseVector(breeze.numerics.sin.sinDoubleImpl(deg)))
  val testY: Seq[DenseVector[Double]] = (t until numTest + t).toList.map(deg => DenseVector(breeze.numerics.sin.sinDoubleImpl(deg)))

  val spectralRadius: Double = 0.7
  val sparsity: Double = 0.80


  def main(args: Array[String]): Unit = {

    val esn: EchoStateNetwork = new EchoStateNetworkImplementation(reservoir = new ReservoirImplementation(nInput = 1, nNeurons = 100, sr = spectralRadius, sp = sparsity, activation = new Sigmoid), readout = new LinearRegression(reg = 1, c = 1))
    esn.fit(trainX, trainY)

    val predictedTestY: Seq[DenseVector[Double]] = testX.map(xi => esn.transform(x = xi))
    val gionas = (testY zip predictedTestY).flatMap(yVEP => (yVEP._1.data zip yVEP._2.data).map(y => (y._1 - y._2) * (y._1 - y._2))).sum / testY.length
    println(s"MSE $gionas")

  }
}
