import com.lucarc.scalaesn.ESN
import com.lucarc.scalaesn.EchoStateNetwork
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.{FeatureSpec, GivenWhenThen}

class ESNTest extends FeatureSpec with GivenWhenThen {
  info(
    """
      |
    """.stripMargin)
  feature("Reservoir Construction") {
    scenario("Given seed, spectral radius, sparsity reservoir is constructed.") {

      Given(
        """
          |
        """.stripMargin)
      val nNeurons: Int = 10
      val nInput: Int = 3
      val spectralRadius: Double = 0.75
      val sparsity: Double = 0.75



      val esn: ESN = new EchoStateNetwork(nInput = nInput, nNeurons = nNeurons, sr = spectralRadius, sp = sparsity)

      When("Network is created.")

      Then(s"Reservoir is a $nNeurons x $nNeurons matrix, Win is a $nInput X $nNeurons matrix.")
      assertResult(nNeurons)(esn.reservoir.rows)
      assertResult(nNeurons)(esn.reservoir.cols)
      assertResult(nInput)(esn.inputLayer.rows)
      assertResult(nNeurons)(esn.inputLayer.cols)

      val epsilon = 1e-4f
      implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(epsilon)
      assert(spectralRadius===breeze.linalg.max(breeze.linalg.eig(esn.reservoir).eigenvalues))


    }

  }

}