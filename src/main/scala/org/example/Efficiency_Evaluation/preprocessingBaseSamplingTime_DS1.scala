package org.example.Efficiency_Evaluation

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.SonPartitionCoalescer
import org.apache.spark.sql.RspContext.NewRDDFunc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{Partitioner, SparkConf, ml}
import org.joda.time.DateTime
import org.example.functions_lib.{MyPartitioner,sampling_Without_Replacement,obtainList}
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object preprocessingBaseSamplingTime_DS1 {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("calculate_maxksdistance_higgs")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    val DS1_DF: DataFrame = spark.read.parquet("Dataset/DS1")
    println("CDFRS")
    val now_CDFRS1: Date=new Date()
    val DS1_bigsample = DS1_DF.rdd.coalesce(738, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(8192, 738).toArray)))
    val DS1_RSP = DS1_bigsample.toRSP(DS1_bigsample.getNumPartitions).map(a=>(a.getInt(0),a.getAs[linalg.Vector](1)))
    var b = 5 //number of RSP blocks in sampling
    val DS1_sample = DS1_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(737, b).toArray)))
    DS1_sample.toDF().show()
    val now_CDFRS2: Date=new Date()
    val CDFRS=now_CDFRS2.getTime - now_CDFRS1.getTime
    println("CDFRS_SamplingTime："+CDFRS)
}
}
