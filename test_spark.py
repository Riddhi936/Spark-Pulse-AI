from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SparkPulseAI") \
    .master("local[*]") \
    .getOrCreate()

print("================================")
print("SparkPulse AI")
print("Spark started successfully!")
print("Spark version:", spark.version)
print("================================")

spark.stop()