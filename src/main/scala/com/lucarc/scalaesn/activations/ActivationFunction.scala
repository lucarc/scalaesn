package com.lucarc.scalaesn.activations

import shapeless.T

trait ActivationFunction {
  def activate[T](x: T): T

}
