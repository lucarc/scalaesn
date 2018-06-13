import com.lucarc.scalaesn.{EchoStateNetwork, EchoStateNetworkImplementation}
import com.lucarc.scalaesn.activations.Sigmoid
import com.lucarc.scalaesn.layers.ReservoirImplementation
import com.lucarc.scalaesn.readouts.implementation.LinearRegression
import org.scalatest.{FeatureSpec, GivenWhenThen}

class LoadRestoreTest extends FeatureSpec with GivenWhenThen {
  info(
    """
      |
    """.stripMargin)
  feature("CSC Test") {
    scenario("Testing constructor for CSC") {

      Given(
        """
          |
        """.stripMargin)
      val nNeurons: Int = 10
      val nInput: Int = 3
      val nSamples: Int = 20
      val spectralRadius: Double = 0.75
      val sparsity: Double = 0.75

      val esn: EchoStateNetwork =  new EchoStateNetworkImplementation(reservoir = new ReservoirImplementation(nInput = nInput, nNeurons = nNeurons, sr = spectralRadius, sp = sparsity, activation = new Sigmoid), readout = new LinearRegression(reg = 1, c = 1))
      esn.dump("./dump.esn")


    }
  }
}
