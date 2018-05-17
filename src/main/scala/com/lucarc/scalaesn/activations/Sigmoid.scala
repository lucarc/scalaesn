package com.lucarc.scalaesn.activations

class Sigmoid extends ActivationFunction {
  override def activate(x: Double): Double = {
    1f / (1f - Math.exp(-x))
  }
}
