package com.lucarc.scalaesn.activations

import java.lang

class Sigmoid extends ActivationFunction {
  override def activate(x: Double): Double = {
    lang.Double.max(1f / (1f - Math.exp(-x)), 1e-5)
  }
}
