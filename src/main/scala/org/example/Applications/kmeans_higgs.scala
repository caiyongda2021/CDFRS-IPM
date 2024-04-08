package org.example.Applications

import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.RspContext.NewRDDFunc
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.row_number
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.example.functions_lib.{MyPartitioner,sampling_Without_Replacement}
import java.util.Date
import scala.collection.mutable.ArrayBuffer

object kmeans_higgs {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("kmeans")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    import spark.implicits._
    //read and prepare sampling data
    val higgs: DataFrame = spark.read.option("inferSchema", "true").parquet("/user/caiyongda/realWorldDataset/HIGGS")
    val higgs_LabeledPoint: RDD[LabeledPoint] = higgs.rdd.map(a => LabeledPoint(a.getDouble(0), a.getAs[linalg.Vector](1)))
    val higgs_LabeledPoint_DF: DataFrame = higgs_LabeledPoint.toDF("label","features")
    val spec = Window.partitionBy().orderBy($"label")
    val higgs_LabeledPoint_DF_id: DataFrame = higgs_LabeledPoint_DF.withColumn("id", row_number().over(spec))
    val records: Long = higgs_LabeledPoint_DF_id.count()
    val higgs_fulldataset = higgs_LabeledPoint_DF_id.rdd.map(k => (k.getInt(2), (k.getDouble(0), k.getAs[linalg.Vector](1)))).partitionBy(new MyPartitioner(8004, records / 8000)).map(k => k._2)
    higgs_fulldataset.toDF().show()
    val now_CDFRS1total: Date=new Date()
    val higgs_big_sample = higgs_fulldataset.coalesce(738, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(8000, 738).toArray)))
    val higgs_big_sample_RSP: RspRDD[(Double, linalg.Vector)] = higgs_big_sample.toRSP(higgs_big_sample.getNumPartitions)
    val arr = new ArrayBuffer[Double]()
    var i = 1
    var b: Int = 1
    //k-means clustering
    while(i<21)
      {
        val higgs_CDFRS_sample = higgs_big_sample_RSP.coalesce(b, false, Option(new SonPartitionCoalescer(sampling_Without_Replacement(737, b).toArray)))
        val kmeans = new KMeans().setK(args(0).toInt).setSeed(1L)
        val model = kmeans.fit(higgs_CDFRS_sample.toDF("label","features"))
        val predictions = model.transform(higgs_fulldataset.toDF("label","features"))
        // Evaluate clustering by computing Silhouette score
        val evaluator = new ClusteringEvaluator()
        val silhouette = evaluator.evaluate(predictions)
        arr += silhouette
        i = i+1
      }
    val now_CDFRS2total: Date=new Date()
    val CDFRS_timetotal=(now_CDFRS2total.getTime -now_CDFRS1total.getTime)/20
    println("The Silhouette score is (1 CDFRS block):"+arr)
    println("running timetotal:"+ CDFRS_timetotal)
  }
}
