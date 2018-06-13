package com.lucarc.scalaesn.activations

@SerialVersionUID(103L)
trait ActivationFunction extends Serializable{
  def activate(x: Double): Double

}
