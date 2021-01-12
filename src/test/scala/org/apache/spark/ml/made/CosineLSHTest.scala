package org.apache.spark.ml.made

import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

class CosineLSHTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.0001
  lazy val data: DataFrame = CosineLSHTest._data
  lazy val vectors: Seq[Vector] = CosineLSHTest._vectors
  lazy val hyperplanes: Array[Vector] = CosineLSHTest._hyperplanes

  "Model" should "test hash function" in {
    val cosineLSHModel: CosineLSHModel = new CosineLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")
    val testVector = linalg.Vectors.fromBreeze(breeze.linalg.Vector(3, 4, 5, 6))
    val sketch = cosineLSHModel.hashFunction(testVector)

    sketch.length should be(3)
    sketch(0)(0) should be(1.0)
    sketch(1)(0) should be(1.0)
    sketch(2)(0) should be(-1.0)
  }

  "Model" should "test hash distance" in {
    val cosineLSHModel: CosineLSHModel = new CosineLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")
    val testVector1 = linalg.Vectors.fromBreeze(breeze.linalg.Vector(3, 4, 5, 6))
    val sketch1 = cosineLSHModel.hashFunction(testVector1)
    val testVector2 = linalg.Vectors.fromBreeze(breeze.linalg.Vector(4, 3, 2, 1))
    val sketch2 = cosineLSHModel.hashFunction(testVector2)
    val similarity = cosineLSHModel.hashDistance(sketch1, sketch2)
    sketch1.foreach(println)
    sketch2.foreach(println)

    similarity should be ((1.0/3.0) +- delta)
  }

  "Model" should "test key distance" in {
    val cosineLSHModel: CosineLSHModel = new CosineLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")
    val testVector1 = linalg.Vectors.fromBreeze(breeze.linalg.Vector(3, 4, 5, 6))
    val testVector2 = linalg.Vectors.fromBreeze(breeze.linalg.Vector(4, 3, 2, 1))
    val keyDistance = cosineLSHModel.keyDistance(testVector1, testVector2)

    keyDistance should be(1 - 0.7875 +- delta)
  }

  "Model" should "transform data" in {
    val cosineLSH: CosineLSH = new CosineLSH(
    ).setNumHashTables(2)
      .setInputCol("features")
      .setOutputCol("hashes")
    val model = cosineLSH.fit(data)
    val transformedData = model.transform(data)

    transformedData.count() should be(3)
  }

  "Model" should "approx similarity join" in {
    val model: CosineLSHModel = new CosineLSHModel(
      randHyperPlanes = hyperplanes
    ).setInputCol("features")
      .setOutputCol("hashes")

    val approxData = model.approxSimilarityJoin(data, data, 1)
    approxData.count() should be(9)
  }
}


object CosineLSHTest extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(1, 2, 3, 4),
    Vectors.dense(5, 4, 9, 7),
    Vectors.dense(9, 6, 4, 5)
  )

  lazy val _hyperplanes = Array(
    Vectors.dense(1, -1, 1, 1),
    Vectors.dense(-1, 1, -1, 1),
    Vectors.dense(1, 1, -1, -1)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
}
