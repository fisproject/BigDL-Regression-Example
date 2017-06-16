package com.intel.analytics.bigdl.example

import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.types.{StructType, StructField, DoubleType}

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.optim._
// import com.intel.analytics.bigdl.numeric.NumericFloat // be careful
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.dataset._

object RegressionExample {

  def loadData(path: String, schema: StructType): DataFrame = {
    val spark = SparkSession.builder.appName("").getOrCreate()
    val df = spark.read.option("header", true).schema(schema).csv(path)
    df
  }

  def transform(df: DataFrame, feats: Array[String]): DataFrame = {
      val assembler = new VectorAssembler().setInputCols(feats).setOutputCol("features")
      val transformedDF = assembler.transform(df)
      val scaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(false)
      val scalerModel = scaler.fit(transformedDF)
      val scaledDF = scalerModel.transform(transformedDF)
      scaledDF
  }

  def convertToSampleRDD(df: DataFrame): RDD[Sample[Float]] = {
    df.rdd.map {case row: org.apache.spark.sql.Row =>
      val vec  = row(6).asInstanceOf[DenseVector]
      val feats = Array(vec(0).toFloat , vec(1).toFloat, vec(2).toFloat, vec(3).toFloat)
      val label = row(0).toString.toFloat
      Sample(
          featureTensor = Tensor(feats, Array(1, 4)).contiguous(),
          labelTensor = Tensor(Array(label), Array(1))
       )
    }
  }

  def main(args: Array[String]): Unit = {
      val conf = Engine.createSparkConf()
        .setAppName("SimpleRegression")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)

      Engine.init

      val schema = StructType(
          StructField("Ozone", DoubleType, true) ::
          StructField("Solar", DoubleType, true) ::
          StructField("Wind", DoubleType, true) ::
          StructField("Temp", DoubleType, true) ::
          StructField("Month", DoubleType, true) ::
          StructField("Day", DoubleType, true) :: Nil
      )

      val df = loadData("data/airquality.csv", schema)
      val scaledDF = transform(df, Array("Solar", "Wind", "Temp", "Month"))
      scaledDF.show

      val sampleRDD = convertToSampleRDD(scaledDF)
      val trainingSplit = 0.7
      val Array(trainingRDD, validationRDD) = sampleRDD.randomSplit(Array(trainingSplit, 1 - trainingSplit))
      println(trainingRDD.count)
      println(validationRDD.count)

      val batchSize = 2
      val optimizer = Optimizer(
          model = SimpleRegression(),
          sampleRDD = trainingRDD,
          criterion = new MSECriterion[Float](),
          batchSize = batchSize
      )

      val optimMethod = new RMSprop[Float](learningRate = 0.01)

      val result = optimizer
        .setOptimMethod(optimMethod)
        .setValidation(
            Trigger.everyEpoch,
            validationRDD,
            Array(), // new Loss[Float]
            batchSize)
        .setEndWhen(Trigger.maxEpoch(30))
        .optimize()

      result.predict(validationRDD).collect.foreach(println)
      // validationRDD.foreach(s => println(s.label))
      // result.predict(trainingRDD).collect().foreach(r => println(r.toTensor[Float]))

      sc.stop()
  }
}
