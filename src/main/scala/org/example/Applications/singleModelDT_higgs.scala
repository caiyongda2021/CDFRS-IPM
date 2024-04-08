package org.example.Applications

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg
import org.apache.spark.{Partitioner, SparkConf, ml}
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.SonPartitionCoalescer
import org.apache.spark.sql.RspContext.NewRDDFunc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.example.functions_lib.{MyPartitioner,sampling_Without_Replacement,DT_cal}

object singleModelDT_higgs {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("DT_exp_CDFRS")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    //read and prepare sampling data
    val higgs: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/higgstraindata")
    val higgs_LabeledPoint: RDD[LabeledPoint] = higgs.rdd.map(a => LabeledPoint(a.getDouble(0), a.getAs[linalg.Vector](1)))
    val higgs_LabeledPoint_DF: DataFrame = higgs_LabeledPoint.toDF("label", "features")
    val spec = Window.partitionBy().orderBy($"label")
    val higgs_LabeledPoint_DF_id: DataFrame = higgs_LabeledPoint_DF.withColumn("id", row_number().over(spec))
    val records: Long = higgs_LabeledPoint_DF_id.count()
    val higgs_train_data = higgs_LabeledPoint_DF_id.rdd.map(k => (k.getInt(2), (k.getDouble(0), k.getAs[ml.linalg.Vector](1)))).partitionBy(new MyPartitioner(6002, records / 6000)).map(k => k._2)
    val higgs_test_data: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/higgstestdata")
    higgs_train_data.toDF().show()
    val now_CDFRS1: Date=new Date()
    val higgs_train_data_big_sample = higgs_train_data.coalesce(738, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(6000, 738).toArray)))
    val higgs_train_data_big_sample_RSP = higgs_train_data_big_sample.toRSP(higgs_train_data_big_sample.getNumPartitions)
    val arr = new ArrayBuffer[Double]()
    var i = 1
    var b: Int = 1
    // DT
    while(i<2)
    {
      val higgs_train_data_CDFRS_sample = higgs_train_data_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(737, b).toArray)))
      val result1 = DT_cal(higgs_train_data_CDFRS_sample.toDF("label","features"): DataFrame, higgs_test_data.toDF("label","features"): DataFrame)
      arr += result1
      i = i+1
    }
    val now_CDFRS2: Date=new Date()
    val CDFRS_time=(now_CDFRS2.getTime -now_CDFRS1.getTime)
    println("The final accuracy is(1 block):"+arr)
    println("running time:"+ CDFRS_time)
  }
}
