import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.lucarc.scalaesn.activations.Sigmoid
import com.lucarc.scalaesn.{EchoStateNetwork, EchoStateNetworkImplementation}
import com.lucarc.scalaesn.layers.{Reservoir, ReservoirImplementation}
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


      val reservoir: Reservoir = new ReservoirImplementation(nInput = nInput, nNeurons = nNeurons, sr = spectralRadius, sp = sparsity, activation = new Sigmoid)

      When("Network is created.")

      Then(s"Reservoir is a $nNeurons x $nNeurons matrix, Win is a $nInput X $nNeurons matrix.")
      assertResult(nNeurons)(reservoir.reservoir.rows)
      assertResult(nNeurons)(reservoir.reservoir.cols)
      assertResult(nInput)(reservoir.inputLayer.rows)
      assertResult(nNeurons)(reservoir.inputLayer.cols)

      val epsilon = 1e-4f
      implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(epsilon)
      assert(breeze.linalg.max(breeze.linalg.eig(reservoir.reservoir).eigenvalues) === spectralRadius)

      val x: DenseMatrix[Double] = DenseMatrix.rand(1, 3)

    }

  }
  feature("Dataset Slicing") {
    scenario("Dataset is divided by rows for multimensional vectors") {

      Given(
        """
          |
        """.stripMargin)
      val nNeurons: Int = 10
      val nInput: Int = 3
      val nSamples: Int = 20
      val spectralRadius: Double = 0.75
      val sparsity: Double = 0.75


      val samples: Seq[DenseVector[Double]] = {0 until nSamples}.map(i => DenseVector.ones[Double](nInput))
      assert(nSamples == samples.size)
    }


  }

  feature("Reservoir Activation") {
    scenario("Reservoir is constructed and activated successfully") {

      Given(
        """
          |
        """.stripMargin)
      val nNeurons: Int = 10
      val nInput: Int = 3
      val nSamples: Int = 20
      val spectralRadius: Double = 0.75
      val sparsity: Double = 0.75


      val reservoir: Reservoir = new ReservoirImplementation(nInput = nInput, nNeurons = nNeurons, sr = spectralRadius, sp = sparsity, activation = new Sigmoid)
      val esn: EchoStateNetwork = new EchoStateNetworkImplementation(reservoir = reservoir, readout = null)

      When("Network is created.")

      Then(s"Reservoir is a $nNeurons x $nNeurons matrix, Win is a $nInput X $nNeurons matrix.")

      val samples: Seq[DenseVector[Double]] = {0 until nSamples}.map(i => DenseVector.ones[Double](nInput))
      val v: Seq[DenseVector[Double]] = samples.map(sample => reservoir.activate(x = sample))
      assertResult(nSamples)(v.length)
      v.foreach(vec => assertResult(nNeurons)(vec.length))

    }


  }
}
