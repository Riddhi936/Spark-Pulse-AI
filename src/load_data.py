from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("SparkPulse AI - Dataset Loading") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "data/social_media.csv",
    header=True,
    inferSchema=True
)

# Display basic information
print("\n========== DATASET LOADED SUCCESSFULLY ==========")

print("\nNumber of Rows:", df.count())

print("\nColumns:")
print(df.columns)

print("\nSchema:")
df.printSchema()

print("\nFirst 5 Rows:")
df.show(5, truncate=False)

# Stop Spark
spark.stop()