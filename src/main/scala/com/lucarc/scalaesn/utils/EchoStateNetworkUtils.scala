package com.lucarc.scalaesn.utils

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import com.lucarc.scalaesn.EchoStateNetwork

object EchoStateNetworkUtils {


  //TODO: to be implemented correctly
  def dump(echoStateNetwork: EchoStateNetwork, filepath: String): Unit = {
    // (2) write the instance out to a file
    val oos = new ObjectOutputStream(new FileOutputStream(filepath))
    oos.writeObject(this)
    oos.close()
  }

  //TODO: to be implemented correctly
  def load(filepath: String): EchoStateNetwork = {
    // (3) read the object back in
    val ois = new ObjectInputStream(new FileInputStream(filepath))
    val esn = ois.readObject.asInstanceOf[EchoStateNetwork]
    ois.close()
    esn
  }
}
