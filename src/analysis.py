from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    lower,
    regexp_replace,
    to_date,
    count,
    avg
)

spark = SparkSession.builder \
    .appName("SparkPulse AI - Spark Analysis") \
    .master("local[*]") \
    .getOrCreate()

# ============================================================
# 1. LOAD DATA
# ============================================================

df = spark.read.csv(
    "data/social_media.csv",
    header=True,
    inferSchema=True
)

print("\n========== DATA LOADED ==========")
print("Original Rows:", df.count())

# ============================================================
# 2. PREPROCESS DATA
# ============================================================

df = df.drop("_c0", "Unnamed: 0")

df = df.dropDuplicates()

df = df.withColumn(
    "Text",
    trim(
        regexp_replace(
            lower(col("Text")),
            r"[^a-zA-Z0-9\s]",
            ""
        )
    )
)

df = df.withColumn(
    "Sentiment",
    trim(col("Sentiment"))
)

df = df.withColumn(
    "Platform",
    trim(col("Platform"))
)

df = df.withColumn(
    "Country",
    trim(col("Country"))
)

df = df.withColumn(
    "Date",
    to_date(col("Timestamp"))
)

print("Preprocessed Rows:", df.count())

# ============================================================
# 3. SENTIMENT / EMOTION DISTRIBUTION
# ============================================================

print("\n========== SENTIMENT / EMOTION DISTRIBUTION ==========")

sentiment_analysis = df.groupBy("Sentiment") \
    .count() \
    .orderBy(col("count").desc())

sentiment_analysis.show(20, truncate=False)

# ============================================================
# 4. PLATFORM ANALYSIS
# ============================================================

print("\n========== PLATFORM ANALYSIS ==========")

platform_analysis = df.groupBy("Platform") \
    .count() \
    .orderBy(col("count").desc())

platform_analysis.show(truncate=False)

# ============================================================
# 5. COUNTRY ANALYSIS
# ============================================================

print("\n========== COUNTRY ANALYSIS ==========")

country_analysis = df.groupBy("Country") \
    .count() \
    .orderBy(col("count").desc())

country_analysis.show(10, truncate=False)

# ============================================================
# 6. ENGAGEMENT ANALYSIS
# ============================================================

print("\n========== PLATFORM ENGAGEMENT ==========")

engagement_analysis = df.groupBy("Platform").agg(
    count("*").alias("Posts"),
    avg("Likes").alias("Average_Likes"),
    avg("Retweets").alias("Average_Retweets")
)

engagement_analysis.show(truncate=False)

# ============================================================
# 7. HOURLY ACTIVITY
# ============================================================

print("\n========== HOURLY ACTIVITY ==========")

hourly_analysis = df.groupBy("Hour") \
    .count() \
    .orderBy(col("count").desc())

hourly_analysis.show(24)

# ============================================================
# 8. SPARK SQL ANALYSIS
# ============================================================

print("\n========== SPARK SQL ANALYSIS ==========")

# Create temporary SQL table
df.createOrReplaceTempView("social_media")

# Query 1: Posts by Platform
platform_sql = spark.sql("""
    SELECT Platform, COUNT(*) AS Total_Posts
    FROM social_media
    GROUP BY Platform
    ORDER BY Total_Posts DESC
""")

print("\n--- Posts by Platform ---")
platform_sql.show()

# Query 2: Posts by Country
country_sql = spark.sql("""
    SELECT Country, COUNT(*) AS Total_Posts
    FROM social_media
    GROUP BY Country
    ORDER BY Total_Posts DESC
""")

print("\n--- Posts by Country ---")
country_sql.show(10)

# Query 3: Most Active Hours
hour_sql = spark.sql("""
    SELECT Hour, COUNT(*) AS Total_Posts
    FROM social_media
    GROUP BY Hour
    ORDER BY Total_Posts DESC
""")

print("\n--- Most Active Hours ---")
hour_sql.show(10)

# ============================================================
# 9. EMOTION CLASS CHECK
# ============================================================

print("\n========== EMOTION CLASS CHECK ==========")

emotion_count = df.select("Sentiment").distinct().count()

print("Total Unique Emotion/Sentiment Classes:", emotion_count)

print("\nAll Emotion/Sentiment Classes:")
df.select("Sentiment") \
    .distinct() \
    .orderBy("Sentiment") \
    .show(200, truncate=False)

# ============================================================
# STOP SPARK
# ============================================================

spark.stop()

print("\n========== SPARK ANALYSIS COMPLETED ==========")