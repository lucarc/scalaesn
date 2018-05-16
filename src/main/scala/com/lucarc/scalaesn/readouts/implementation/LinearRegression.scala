package com.lucarc.scalaesn.readouts.implementation

import breeze.linalg.DenseMatrix
import com.lucarc.scalaesn.readouts.Readout

class LinearRegression extends Readout {



  override def apply(x: DenseMatrix[Double]): Any = ???

  override def train(x: DenseMatrix[Double]): Unit = ???
}
