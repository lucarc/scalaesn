package com.lucarc.scalaesn.activations
import shapeless.T

class Sigmoid extends ActivationFunction {
  override def activate(x: T): Any = {
    x
  }

}
