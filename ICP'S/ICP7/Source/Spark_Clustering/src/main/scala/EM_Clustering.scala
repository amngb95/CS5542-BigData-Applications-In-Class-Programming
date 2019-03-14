import java.io.PrintStream

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.clustering.GaussianMixture
object EM_Clustering {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\Mayanka Lenevo F Drive\\winutils")
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val features=sc.textFile("data\\Flickr8k.token.txt")
      .map(f=>{
        val str=f.replaceAll(",","")
        val ff=f.split(" ")
        ff.drop(1).toSeq
      })
    val hashingTF=new HashingTF(100)

    val tf=hashingTF.transform(features)

    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)


    // Cluster the data into two classes using GaussianMixture
    val gmm = new GaussianMixture().setK(10).run(tf)

    val clusters=gmm.predict(tf)

    val out=new PrintStream("data\\resultsEM.csv")

    features.zip(clusters).collect().foreach(f=>{
      out.println(f._1.mkString(" ")+","+f._2)
    })

  }

}
