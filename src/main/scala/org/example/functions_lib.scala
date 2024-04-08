package org.example

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.{Partitioner, SparkConf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object functions_lib {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("yarn").setAppName("function_lib")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
  }
  class MyPartitioner(val num:Int,val records:Long) extends Partitioner {
    override def numPartitions: Int = num
    override def getPartition(key: Any): Int = {
      val len = (key.toString.toInt/records).toInt
      len
    }
  }

  def sampling_Without_Replacement(total: Int, subNum: Int) = {
    var arr = 0 to total toArray
    var outList: List[Int] = Nil
    var border = arr.length
    for (i <- 0 to subNum - 1) {
      val index = (new Random).nextInt(border)
      outList = outList ::: List(arr(index))
      arr(index) = arr.last
      arr = arr.dropRight(1)
      border -= 1
    }
    outList
  }

  def obtainList(start: Int, end: Int) = {
    var arr = start to end-1 toArray
    var outList: List[Int] = Nil
    val border = arr.length
    var i = 0
    while(i<border){
      outList = outList ::: List(arr(i))
      i = i+1
    }
    outList
  }

  def Map1(v:Double,t:Double): Int ={
    var ind = 0
    if(v<t) {
      ind = 1
    }
    ind
  }

  def Map2(v:Double,t:Double): Int = {
    val ind = v / t - (v / t).toInt
    ind.toInt
  }

  def ksDistance(sample1: RDD[Double], sample2: RDD[Double]): Double = {
    val n1 = sample1.count().toDouble
    val n2 = sample2.count().toDouble
    val rdd21: RDD[Double] = sample1.union(sample2).map(k => (k, 1)).sortByKey().zipWithIndex().reduceByKey((_, v) => v).map(k => k._2.toDouble).coalesce(1).map(k => (k, 1)).sortByKey().map(k => k._1+1)
    val rddtotal: RDD[Double] = sample2.map(k => (k, 1)).sortByKey().zipWithIndex().reduceByKey((_, v) => v).map(k => k._2.toDouble).coalesce(1).map(k=>(k,1)).sortByKey().map(k=>k._1+1)
    val rddfinal = rdd21.zip(rddtotal).map(k => (k._1 - k._2,k._2))
    val ksdistance: Double = rddfinal.map(k => math.abs(k._1/n1 - k._2/n2)).max()
    ksdistance
  }

  def DT_cal(trainingData: DataFrame, testData: DataFrame): Double = {
    val classifier = new DecisionTreeClassifier()
    val model: DecisionTreeClassificationModel = classifier.fit(trainingData)
    val result: DataFrame = model.transform(testData)
    val evaluator4 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val result1 = evaluator4.evaluate( result.select("label", "features","prediction"))
    result1
  }

  def LR_cal(trainingData: DataFrame, testData: DataFrame): Double ={
    //val Array(trainingData, testData) = training.randomSplit(Array(0.3, 0.7))
    val classifier = new LogisticRegression()
    val ovrModel = classifier.fit(trainingData)
    val result = ovrModel.transform(testData)
    val evaluator4 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val result1 = evaluator4.evaluate( result.select("label", "features","prediction"))
    result1
  }

  def RF_cal(trainingData: DataFrame, testData: DataFrame): Double ={

    val rf = new RandomForestClassifier()
      .setNumTrees(5)
    val model = rf.fit(trainingData)
    val predictions = model.transform(testData)
    val predictionAndLabels = predictions.select("prediction","label")
    val evaluator4 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val result = evaluator4.evaluate(predictionAndLabels)
    result
  }

  def DT_calOriginal(trainingData: DataFrame, testData: DataFrame): DataFrame = {
    val classifier = new DecisionTreeClassifier()
    val model: DecisionTreeClassificationModel = classifier.fit(trainingData)
    val result: DataFrame = model.transform(testData)
    //val evaluator4 = new MulticlassClassificationEvaluator()
    //  .setMetricName("accuracy")
    //val result1 = evaluator4.evaluate( result.select("label", "features","prediction"))
    val result1: DataFrame = result.select("prediction")
    result1
  }

  def EQH_cal(data:DataFrame,K:Int): Array[Double]={
    val rdd4: RDD[Double] = data.orderBy("value").rdd.map(k => k.getDouble(0))
    val data_num = (data.count()/K).toInt
    var i = 1
    val arr = new ArrayBuffer[Double]()
    val index: Double = rdd4.take(1).max
    //val index: Double = getLastElement(rdd4.take(1).to)
    arr += index
    while(i<K)
    {
      //val index: Double = rdd4.take(i * data_num).max
      val index: Double = rdd4.take(i * data_num).reduce((a,b)=>math.max(a,b))
      arr += index
      i = i+1
    }
    //arr += rdd4.max
    arr += rdd4.reduce((a,b)=>math.max(a,b))
    arr.toArray
  }

  def Vote(input:(Array[Array[Int]],Int))={
    val numOfModel: Int = input._1.length
    val labels: Array[Array[Int]] = input._1
    val numOfFeatures: Int = input._2
    val result = new Array[Int](numOfFeatures)
    val map: mutable.Map[Int, Int] = mutable.Map[Int,Int]()
    for(i <- 0 until numOfFeatures){
      for(m <- 0 until numOfModel){
        map.put(labels(m)(i),map.getOrElse(labels(m)(i),0)+1)
      }
      result(i)=map.maxBy(_._2)._1
      map.clear()
    }
    result
  }

  def getLastElement[T](rdd:RDD[T]):T={
    rdd.reduce((_,y)=>y)
  }

  def AS_cal(data:DataFrame,split_num:Int,K:Int): Double={
    val rdd4: (Array[Double], Array[Long]) = data.rdd.map(k => k.getDouble(0)).histogram(split_num)
    val data_num = data.count()/K
    //println(data_num)
    var i = 0
    var j = 0
    var temp:Long = 0
    val arr = new ArrayBuffer[Double]()
    while(i<K&j<split_num)
    {
      while(temp<data_num&j<split_num)
      {
        val arr: Array[Long] = rdd4._2.toArray
        temp = temp + arr(j)
        j = j+1
      }
      arr += temp-data_num
      temp = 0
      i = i+1
    }
    val rddmax: Double = arr.max
    rddmax
  }
}
