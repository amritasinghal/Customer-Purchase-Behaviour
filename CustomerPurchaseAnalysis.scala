import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

val spark = SparkSession.builder()
  .appName("Customer Purchase Behaviour Analysis")
  .master("local[*]")
  .getOrCreate()

// Load Dataset
val retailDF = spark.read
  .option("header","true")
  .option("inferSchema","true")
  .csv("/Users/amritasinghal/Desktop/online_retail_II.csv")

// Show Original Data
retailDF.show(5)
retailDF.columns.foreach(println)

// Data Cleaning
val cleanedDF = retailDF.na.drop().dropDuplicates()

// Show Cleaned Data
cleanedDF.show(5)

// Count Rows After Cleaning
println("Total Rows After Cleaning: " + cleanedDF.count())
// Create TotalAmount Column
val finalDF = cleanedDF.withColumn("TotalAmount", col("Quantity") * col("Price"))

finalDF.show(5)

// Customer Spending Analysis
val customerDF = finalDF.groupBy("Customer ID")
  .agg(
    sum("TotalAmount").alias("TotalSpent"),
    count("Invoice").alias("Frequency")
  )

customerDF.show(10)

// Feature Vector Creation
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(Array("TotalSpent", "Frequency"))
  .setOutputCol("features")

val featureDF = assembler.transform(customerDF)

featureDF.select("Customer ID", "features").show(10)

// K-Means Clustering
import org.apache.spark.ml.clustering.KMeans

val kmeans = new KMeans()
  .setK(3)
  .setSeed(1L)
  .setFeaturesCol("features")
  .setPredictionCol("Cluster")

val model = kmeans.fit(featureDF)

val clusteredDF = model.transform(featureDF)

clusteredDF.select("Customer ID", "TotalSpent", "Frequency", "Cluster").show(20)
// Show Cluster Centers
println("Cluster Centers:")
model.clusterCenters.foreach(println)
import org.apache.spark.sql.functions.when

val labeledDF = clusteredDF.withColumn(
  "CustomerSegment",
  when(col("Cluster") === 1, "Premium Customers")
    .when(col("Cluster") === 2, "Regular Customers")
    .otherwise("Medium Customers")
)

labeledDF.select("Customer ID", "TotalSpent", "Frequency", "CustomerSegment").show(20)