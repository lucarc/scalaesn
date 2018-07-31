import breeze.linalg.DenseVector
import org.scalatest.{FeatureSpec, GivenWhenThen}
import breeze.linalg.{DenseMatrix, DenseVector}

class DataFormatting extends FeatureSpec with GivenWhenThen {
  info(
    """
      |
    """.stripMargin)
  feature("Data Formatting ") {
    scenario("Convert a sequence of vectors into a dense matrix.") {

      val sequence: Seq[DenseVector[Double]] = (0 until 20).map(i => DenseVector.ones[Double](5))
      val matrix = DenseMatrix(sequence.flatMap(_.data)).reshape(sequence.length, sequence.head.length)

      assert(matrix.isInstanceOf[DenseMatrix[Double]])





    }
  }
}
