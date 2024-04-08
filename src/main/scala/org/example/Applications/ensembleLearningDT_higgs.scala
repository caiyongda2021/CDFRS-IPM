package org.example.Applications

import org.apache.spark.{SparkConf, ml}
import org.apache.spark.ml.{linalg}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.RspContext.NewRDDFunc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.example.functions_lib.{MyPartitioner,sampling_Without_Replacement,obtainList,DT_calOriginal,Vote}

object ensembleLearningDT_higgs {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("DT_ensemble_exp_CDFRS")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    val higgs: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/higgstraindata")
    val higgs_LabeledPoint: RDD[LabeledPoint] = higgs.rdd.map(a => LabeledPoint(a.getDouble(0), a.getAs[linalg.Vector](1)))
    val higgs_LabeledPoint_DF: DataFrame = higgs_LabeledPoint.toDF("label", "features")
    val spec = Window.partitionBy().orderBy($"label")
    val higgs_LabeledPoint_DF_id: DataFrame = higgs_LabeledPoint_DF.withColumn("id", row_number().over(spec))
    val records: Long = higgs_LabeledPoint_DF_id.count()
    val higgs_train_data = higgs_LabeledPoint_DF_id.rdd.map(k => (k.getInt(2), (k.getDouble(0), k.getAs[ml.linalg.Vector](1)))).partitionBy(new MyPartitioner(6002, records / 6000)).map(k => k._2)
    val higgs_test_data: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/higgstestdata")
    val b: Int = args(0).toInt
    val higgs_train_data_big_sample = higgs_train_data.coalesce(738, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(6000, 738).toArray)))
    val higgs_train_data_big_sample_RSP = higgs_train_data_big_sample.toRSP(higgs_train_data_big_sample.getNumPartitions)
    //ensemble learning (DT)
    var i = 1
    val higgs_CDFRS_sample1 = higgs_train_data_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(obtainList((i-1)*b+1,i*b+1).toArray)))
    val result1 = DT_calOriginal(higgs_CDFRS_sample1.toDF("label","features"): DataFrame, higgs_test_data.toDF("label","features"): DataFrame)
      .rdd.map(a => a.getDouble(0).toInt).collect()
    i = i+1
    val higgs_CDFRS_sample2 = higgs_train_data_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(obtainList((i-1)*b+1,i*b+1).toArray)))
    val result2 = DT_calOriginal(higgs_CDFRS_sample2.toDF("label","features"): DataFrame, higgs_test_data.toDF("label","features"): DataFrame)
      .rdd.map(a => a.getDouble(0).toInt).collect()
    i = i+1
    val higgs_CDFRS_sample3 = higgs_train_data_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(obtainList((i-1)*b+1,i*b+1).toArray)))
    val result3 = DT_calOriginal(higgs_CDFRS_sample3.toDF("label","features"): DataFrame, higgs_test_data.toDF("label","features"): DataFrame)
      .rdd.map(a => a.getDouble(0).toInt).collect()
    i = i+1
    val higgs_CDFRS_sample4 = higgs_train_data_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(obtainList((i-1)*b+1,i*b+1).toArray)))
    val result4 = DT_calOriginal(higgs_CDFRS_sample4.toDF("label","features"): DataFrame, higgs_test_data.toDF("label","features"): DataFrame)
      .rdd.map(a => a.getDouble(0).toInt).collect()
    i = i+1
    val higgs_CDFRS_sample5 = higgs_train_data_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(obtainList((i-1)*b+1,i*b+1).toArray)))
    val result5 = DT_calOriginal(higgs_CDFRS_sample5.toDF("label","features"): DataFrame, higgs_test_data.toDF("label","features"): DataFrame)
      .rdd.map(a => a.getDouble(0).toInt).collect()

    val resultfinal1: Array[Array[Int]] = Array(result1, result2, result3, result4, result5)
    val resultfinal2 = spark.sparkContext.makeRDD(Vote(resultfinal1,higgs_test_data.count().toInt)).map(k=>k.toDouble)
    val rddtestlabel1 = spark.sparkContext.makeRDD(higgs_test_data.select("label").collect()).map(a=>a.getDouble(0))
    val rddtestlabel: DataFrame = resultfinal2.zip(rddtestlabel1).toDF("prediction", "label")
    val evaluator4 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val resultfinal3 = evaluator4.evaluate(rddtestlabel)
    println("ensembleResultDT: "+resultfinal3)
  }
}
