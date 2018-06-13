import breeze.linalg.{DenseVector, SparseVector}
import com.lucarc.scalaesn.activations.Sigmoid
import com.lucarc.scalaesn.layers.{Reservoir, ReservoirImplementation}
import com.lucarc.scalaesn.readouts.implementation.LinearRegression
import com.lucarc.scalaesn.{EchoStateNetwork, EchoStateNetworkImplementation}
import org.scalatest.{FeatureSpec, GivenWhenThen}

class ESNTest extends FeatureSpec with GivenWhenThen {



  feature("Reservoir Activation") {
    scenario("Reservoir is constructed and activated successfully") {

      Given(
        """
          | nNeurons = 10
          | nInput = 3
          | nSamples = 20
          | spectralRadius = 0.75
          | sparsity = 0.75
        """.stripMargin)
      val nNeurons: Int = 10
      val nInput: Int = 3
      val nSamples: Int = 20
      val spectralRadius: Double = 0.75
      val sparsity: Double = 0.75

      val reservoir: Reservoir = new ReservoirImplementation(nInput = nInput, nNeurons = nNeurons, sr = spectralRadius, sp = sparsity, activation = new Sigmoid)

      When("Reservoir is created and fed with 20 samples of 3 doubles")
      val samples: Seq[SparseVector[Double]] = {0 until nSamples}.map(i => SparseVector(DenseVector.ones[Double](nInput).data))
      val v: Seq[SparseVector[Double]] = samples.map(sample => reservoir.activate(x = sample))
      Then(s"Reservoir is activated generating a 20x10 sequence..")
      assertResult(nSamples)(v.length)
      v.foreach(vec => assertResult(nNeurons)(vec.length))
    }
  }

  feature("ESN Activation") {
    scenario("Reservoir is constructed and activated successfully") {

      Given(
        """
          | nNeurons = 10
          | nInput = 3
          | nSamples = 20
          | spectralRadius = 0.75
          | sparsity = 0.75
        """.stripMargin)
      val nNeurons: Int = 10
      val nInput: Int = 3
      val nSamples: Int = 20
      val spectralRadius: Double = 0.75
      val sparsity: Double = 0.75

      val esn: EchoStateNetwork =  new EchoStateNetworkImplementation(reservoir = new ReservoirImplementation(nInput = nInput, nNeurons = nNeurons, sr = spectralRadius, sp = sparsity, activation = new Sigmoid), readout = new LinearRegression(reg = 1, c = 1))

      When("Reservoir is created and fed with 20 samples of 3 doubles")
      val samples: Seq[SparseVector[Double]] = {0 until nSamples}.map(i => SparseVector(DenseVector.ones[Double](nInput).data))
      val yExpected: Seq[SparseVector[Double]] = {0 until nSamples}.map(i => SparseVector(DenseVector.ones[Double](nInput).data))

      Then(s"ESN is trained and no exception is thrown")
      esn.fit(samples, yExpected, 4)
    }
  }
}