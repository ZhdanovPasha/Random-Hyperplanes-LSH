import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, evaluation}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.made.CosineLSH


object searching_params extends App {

  val spark = SparkSession.builder().master("local").getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")
  val df = spark.read
    .option("inferSchema", "true")
    .option("header", "true")
    .csv("src/main/resources/tripadvisor_hotel_reviews.csv")
    .sample(0.1)

  df.show(10)

  val preprocessingPipe = new Pipeline()
    .setStages(Array(
      new RegexTokenizer()
        .setInputCol("Review")
        .setOutputCol("tokenized")
        .setPattern("\\W+"),
      new HashingTF()
        .setInputCol("tokenized")
        .setOutputCol("tf")
        .setNumFeatures(1000),
      new IDF()
        .setInputCol("tf")
        .setOutputCol("tfidf")
    ))

  val Array(train, test) = df.randomSplit(Array(0.8, 0.2))

  val pipe = preprocessingPipe.fit(train)

  val trainFeatures = pipe.transform(train)
  val testFeatures = pipe.transform(test)

  val testFeaturesWithIndex = testFeatures.withColumn("id", monotonicallyIncreasingId())

  var clsh =  new CosineLSH()
    .setInputCol("tfidf")
    .setOutputCol("buckets")
    .setNumHashTables(3)

  var clshModel = clsh.fit(trainFeatures)

  var neighbs = clshModel.approxSimilarityJoin(trainFeatures, testFeaturesWithIndex, 0.7)

  var preds = neighbs
    .withColumn("similarity", lit(1) - col("distCol"))
    .groupBy("datasetB.id")
    .agg((sum(col("similarity") * col("datasetA.Rating")) / sum(col("similarity"))).as("predict"))

  var forMetricValues = testFeaturesWithIndex.join(preds, Seq("id"))
  var finalMetrics = new evaluation.RegressionEvaluator()
    .setLabelCol("Rating")
    .setPredictionCol("predict")
    .setMetricName("rmse")
  println("Current metric")
  println(finalMetrics.evaluate(forMetricValues))

  println("Start grid search")
  val results = Array.range(3, 11, 2).map(numHashes => {

    val clsh = new CosineLSH()
      .setInputCol("tfidf")
      .setOutputCol("buckets")
      .setNumHashTables(numHashes)

    val clshModel = clsh.fit(trainFeatures)

    val thresholds = Array(0.6, 0.7, 0.8).map(threshold => {
      val neighbs = clshModel.approxSimilarityJoin(trainFeatures, testFeaturesWithIndex, threshold)

      val preds = neighbs
        .withColumn("similarity", lit('1') -  col("distCol"))
        .groupBy("datasetB.id")
        .agg(
          (sum(col("similarity") * col("datasetA.Rating")) / sum(col("similarity"))).as("predict"),
          count("datasetA.Rating").as("numNeighbors")
        )

      val forMetricValues = testFeaturesWithIndex.join(preds, Seq("id"))
      val numNeighbors = forMetricValues.select(avg("numNeighbors")).collect.head(0)
      val metric = finalMetrics.evaluate(forMetricValues)

      val res = (numHashes, metric, numNeighbors)
      println(res)
      res
    })
    thresholds
  })

  results.map(x => x.map(y => println(y)))

}
