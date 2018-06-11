package com.lucarc.scalaesn

import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.SparseVector
import com.lucarc.scalaesn.activations.Sigmoid
import com.lucarc.scalaesn.layers.ReservoirImplementation
import com.lucarc.scalaesn.readouts.implementation.LinearRegression
import java.io.File

import scala.collection.parallel.ParSeq

object Main {

  val numTrain: Int = 500000
  val numTest: Int = 1000

  val t: Int = 3

  val trainX: ParSeq[SparseVector[Double]] = (0 until numTrain).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl((deg * scala.math.Pi) / 180))).toParArray
  val trainY: ParSeq[SparseVector[Double]] = (t until numTrain + t).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl((deg * scala.math.Pi) / 180))).toParArray

  val testX: ParSeq[SparseVector[Double]] = (0 until numTest).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl((deg * scala.math.Pi) / 180))).toParArray
  val testY: ParSeq[SparseVector[Double]] = (t until numTest + t).toList.map(deg => SparseVector(breeze.numerics.sin.sinDoubleImpl((deg * scala.math.Pi) / 180))).toParArray

  val spectralRadius: Double = 0.85
  val sparsity: Double = 0.75

  val washout: Int = 150


  def main(args: Array[String]): Unit = {

    val esn: EchoStateNetwork = new EchoStateNetworkImplementation(reservoir = new ReservoirImplementation(nInput = 1, nNeurons = 100, sr = spectralRadius, sp = sparsity, activation = new Sigmoid), readout = new LinearRegression(reg = 1, c = 1))
    esn.fit(x = trainX, yExpected = trainY, numThreads = 4)

    val predictedTestY: ParSeq[SparseVector[Double]] = testX.map(xi => esn.transform(x = xi))
    val unrolledData = (testY zip predictedTestY).flatMap(yVEP => yVEP._1.data zip yVEP._2.data)
    val washedOutData = unrolledData.drop(washout)

    val file: File = new File("file.csv")
    val bw = new BufferedWriter(new FileWriter(file))
    unrolledData.foreach(el => {
      bw.write(el.productIterator.toList.mkString(","))
      bw.write("\n")
    })
    bw.close()

    val mse: Double = washedOutData.map(y => (y._1 - y._2) * (y._1 - y._2)).sum / washedOutData.length
    System.out.print(mse)
    System.exit(0)
  }
}