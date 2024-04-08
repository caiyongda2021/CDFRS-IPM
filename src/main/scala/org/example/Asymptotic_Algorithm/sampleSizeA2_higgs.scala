package org.example.Asymptotic_Algorithm

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.SonPartitionCoalescer
import org.apache.spark.sql.RspContext.NewRDDFunc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{Partitioner, SparkConf, ml}
import org.example.functions_lib.{ksDistance,sampling_Without_Replacement,MyPartitioner}
import java.util.Date
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object sampleSizeA2_higgs {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("Read and save real-world data")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    //read and prepare sampling data
    val higgs: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/HIGGS")
    val higgs_LabeledPoint: RDD[LabeledPoint] = higgs.rdd.map(a => LabeledPoint(a.getDouble(0), a.getAs[linalg.Vector](1)))
    val higgs_LabeledPoint_DF: DataFrame = higgs_LabeledPoint.toDF("label","features")
    val spec = Window.partitionBy().orderBy($"label")
    val higgs_LabeledPoint_DF_id: DataFrame = higgs_LabeledPoint_DF.withColumn("id", row_number().over(spec))
    val records: Long = higgs_LabeledPoint_DF_id.count()
    val higgs_fulldataset = higgs_LabeledPoint_DF_id.rdd.map(k => (k.getInt(2), (k.getDouble(0), k.getAs[ml.linalg.Vector](1)))).partitionBy(new MyPartitioner(8001, records / 8000)).map(k => k._2)
    //A2 algorithm
    val now_CDFRS1=new Date()
    val arr1 = new ArrayBuffer[Double]()
    var l = 0
    while(l<5)
      {
        val higgs_big_sample = higgs_fulldataset.coalesce(738, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(8000, 738).toArray)))
        val higgs_big_sample_RSP = higgs_big_sample.toRSP(higgs_big_sample.getNumPartitions)
        val nums: List[Int] = List(4,12,19,24,26)
        val arr = new ArrayBuffer[Double]()
        var i = 0
        var j = 1
        var flag = 0.0
        var temp = 0
        var distance:Double = 0
        while(j<21)
        {
          while(i<5)
          {
            val higgs_CDFRS_sample_T1 = higgs_big_sample_RSP.map(k => k._2).map(v => v(nums(i).toInt)).coalesce(j.toInt, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(737, j.toInt).toArray)))
            val higgs_CDFRS_sample_T2 = higgs_big_sample_RSP.map(k => k._2).map(v=>v(nums(i).toInt)).coalesce((j+1).toInt, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(737, (j+1).toInt).toArray)))
            distance = ksDistance(higgs_CDFRS_sample_T1, higgs_CDFRS_sample_T2)
            arr += distance
            i = i+1
          }
          j = j+1
          temp = j
          if (arr.max<0.05)
          {
            j=22
          }
          arr.clear()
          i = 0
          flag = 0.0
        }
        arr1+=temp
        l=l+1
        temp=0
      }
    val now_CDFRS2: Date=new Date()
    val CDFRS=now_CDFRS2.getTime -now_CDFRS1.getTime
    println("sampleSizeA2_higgs："+arr1)
    println("sampleSizeA2Time_higgs"+CDFRS/5)
}
}
