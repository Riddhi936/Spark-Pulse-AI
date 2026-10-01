import os
import sys

# Use the same Python environment for Spark workers
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    lower,
    regexp_replace,
    explode,
    split,
    desc,
    avg,
    count
)

# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SparkPulse Trend Detection") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n========== SPARKPULSE TREND DETECTION ==========")

# ============================================================
# 2. LOAD DATA
# ============================================================

file_path = "data/social_media.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Original Rows:", df.count())

# ============================================================
# 3. BASIC PREPROCESSING
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

df = df.withColumn("Sentiment", trim(col("Sentiment")))
df = df.withColumn("Platform", trim(col("Platform")))
df = df.withColumn("Country", trim(col("Country")))

print("Preprocessed Rows:", df.count())

# ============================================================
# 4. PLATFORM TRENDS
# ============================================================

print("\n========== PLATFORM TRENDS ==========")

platform_trends = df.groupBy("Platform") \
    .agg(
        count("*").alias("Total_Posts"),
        avg("Likes").alias("Average_Likes"),
        avg("Retweets").alias("Average_Retweets")
    ) \
    .orderBy(desc("Total_Posts"))

platform_trends.show(truncate=False)

# ============================================================
# 5. COUNTRY TRENDS
# ============================================================

print("\n========== COUNTRY TRENDS ==========")

country_trends = df.groupBy("Country") \
    .agg(
        count("*").alias("Total_Posts"),
        avg("Likes").alias("Average_Likes"),
        avg("Retweets").alias("Average_Retweets")
    ) \
    .orderBy(desc("Total_Posts"))

country_trends.show(10, truncate=False)

# ============================================================
# 6. HOURLY ACTIVITY TRENDS
# ============================================================

print("\n========== HOURLY ACTIVITY TRENDS ==========")

hourly_trends = df.groupBy("Hour") \
    .agg(
        count("*").alias("Total_Posts"),
        avg("Likes").alias("Average_Likes"),
        avg("Retweets").alias("Average_Retweets")
    ) \
    .orderBy(desc("Total_Posts"))

hourly_trends.show(10, truncate=False)

# ============================================================
# 7. SENTIMENT TRENDS
# ============================================================

print("\n========== SENTIMENT TRENDS ==========")

sentiment_trends = df.groupBy("Sentiment") \
    .agg(
        count("*").alias("Total_Posts"),
        avg("Likes").alias("Average_Likes"),
        avg("Retweets").alias("Average_Retweets")
    ) \
    .orderBy(desc("Total_Posts"))

sentiment_trends.show(20, truncate=False)

# ============================================================
# 8. HASHTAG ANALYSIS
# ============================================================

print("\n========== TOP HASHTAGS ==========")

hashtags = df.select("Hashtags") \
    .filter(
        col("Hashtags").isNotNull() &
        (trim(col("Hashtags")) != "")
    )

# Split hashtags using spaces
hashtags = hashtags.withColumn(
    "Hashtag",
    explode(
        split(
            trim(col("Hashtags")),
            r"\s+"
        )
    )
)

hashtags = hashtags.withColumn(
    "Hashtag",
    trim(col("Hashtag"))
)

top_hashtags = hashtags \
    .filter(col("Hashtag") != "") \
    .groupBy("Hashtag") \
    .agg(count("*").alias("Frequency")) \
    .orderBy(desc("Frequency"))

top_hashtags.show(20, truncate=False)

# ============================================================
# 9. TOP ENGAGEMENT POSTS
# ============================================================

print("\n========== TOP ENGAGEMENT POSTS ==========")

engagement = df.withColumn(
    "Engagement",
    col("Likes") + col("Retweets")
)

top_engagement = engagement \
    .select(
        "Text",
        "Platform",
        "Sentiment",
        "Likes",
        "Retweets",
        "Engagement"
    ) \
    .orderBy(desc("Engagement"))

top_engagement.show(10, truncate=False)

# ============================================================
# 10. SPARK SQL TREND ANALYSIS
# ============================================================

print("\n========== SPARK SQL ANALYSIS ==========")

df.createOrReplaceTempView("social_media")

sql_platform = spark.sql("""
    SELECT
        Platform,
        COUNT(*) AS Total_Posts,
        ROUND(AVG(Likes), 2) AS Average_Likes,
        ROUND(AVG(Retweets), 2) AS Average_Retweets
    FROM social_media
    GROUP BY Platform
    ORDER BY Total_Posts DESC
""")

print("\n--- Platform SQL Result ---")
sql_platform.show(truncate=False)

sql_hour = spark.sql("""
    SELECT
        Hour,
        COUNT(*) AS Total_Posts
    FROM social_media
    GROUP BY Hour
    ORDER BY Total_Posts DESC
""")

print("\n--- Hour SQL Result ---")
sql_hour.show(10, truncate=False)

# ============================================================
# 11. TREND SUMMARY
# ============================================================

print("\n========== TREND SUMMARY ==========")

top_platform = platform_trends.first()

top_hour = hourly_trends.first()

top_country = country_trends.first()

top_hashtag = top_hashtags.first()

print("Most Active Platform:", top_platform["Platform"])
print("Most Active Hour:", str(top_hour["Hour"]) + ":00")
print("Top Country by Posts:", top_country["Country"])
print("Top Hashtag:", top_hashtag["Hashtag"])

print("\n========== TREND DETECTION COMPLETED ==========")

spark.stop()