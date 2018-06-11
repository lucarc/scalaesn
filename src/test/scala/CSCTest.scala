import breeze.linalg.CSCMatrix
import breeze.stats.distributions.{Rand, RandBasis}
import org.scalatest.{FeatureSpec, GivenWhenThen}

class CSCTest extends FeatureSpec with GivenWhenThen {
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
      val rand: Rand[Double] = RandBasis.withSeed(42).uniform

      val csc: CSCMatrix[Double] = CSCMatrix.rand(10,10,rand)



    }
  }
}
