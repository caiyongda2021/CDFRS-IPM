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


object preprocessingBaseSamplingTime_higgs {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("calculate_maxksdistance_higgs")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    //read and prepare sampling data
    val higgs: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/HIGGS")
    val higgs_LabeledPoint: RDD[LabeledPoint] = higgs.rdd.map(a => LabeledPoint(a.getDouble(0), a.getAs[linalg.Vector](1)
    ))
    val higgs_LabeledPoint_DF: DataFrame = higgs_LabeledPoint.toDF("label","features")
    val spec = Window.partitionBy().orderBy($"label")
    val higgs_LabeledPoint_DF_id: DataFrame = higgs_LabeledPoint_DF.withColumn("id", row_number().over(spec))
    val records: Long = higgs_LabeledPoint_DF_id.count()
    val higgs_fulldataset = higgs_LabeledPoint_DF_id.rdd.map(k => (k.getInt(2), (k.getDouble(0), k.getAs[ml.linalg.Vector](1)))).partitionBy(new MyPartitioner(8001, records / 8000)).map(k => k._2)

    //CDFRS sampling method
    val now_CDFRS1=new Date()
    println("CDFRS")
    val higgs_big_sample = higgs_fulldataset.coalesce(738, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(8000, 738).toArray)))
    val higgs_big_sample_RSP = higgs_big_sample.toRSP(higgs_big_sample.getNumPartitions)
    val higgs_CDFRS_sample = higgs_big_sample_RSP.coalesce(10, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(782, 10).toArray)))
    higgs_CDFRS_sample.map(k=>k._2).map(v=>(v(0),v(1),v(2),v(3),v(4),v(5),v(6),v(7),v(8),v(9),
      v(10),v(11),v(12),v(13),v(14),v(15),v(16),v(17),v(18),v(19),v(20),v(21)
    )).coalesce(1).toDF().write.csv("datahiggs_temp/CDFRS1")
    higgs_CDFRS_sample.map(k=>k._2).map(v=>(v(22),v(23),v(24),v(25),v(26),v(27)
    )).coalesce(1).toDF().write.csv("datahiggs_temp/CDFRS2")
    val now_CDFRS2: Date=new Date()
    val CDFRS=now_CDFRS1.getTime -now_CDFRS2.getTime
    println("CDFRS_SamplingTime："+CDFRS)
}
}
