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
    count,
    avg,
    desc
)

import matplotlib.pyplot as plt


# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SparkPulse Visualization") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n========== SPARKPULSE VISUALIZATION ==========")


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

chart_folder = "outputs/charts"

os.makedirs(chart_folder, exist_ok=True)

print("Chart folder:", chart_folder)


# ============================================================
# 3. LOAD DATA
# ============================================================

file_path = "data/social_media.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

print("Original Rows:", df.count())


# ============================================================
# 4. PREPROCESS DATA
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
# 5. PLATFORM DISTRIBUTION
# ============================================================

print("\nCreating Platform Distribution chart...")

platform_data = df.groupBy("Platform") \
    .agg(count("*").alias("Total_Posts")) \
    .orderBy(desc("Total_Posts")) \
    .toPandas()

plt.figure(figsize=(8, 5))

plt.bar(
    platform_data["Platform"],
    platform_data["Total_Posts"]
)

plt.title("Social Media Posts by Platform")
plt.xlabel("Platform")
plt.ylabel("Number of Posts")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/platform_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 6. SENTIMENT / EMOTION DISTRIBUTION
# ============================================================

print("Creating Sentiment Distribution chart...")

sentiment_data = df.groupBy("Sentiment") \
    .agg(count("*").alias("Total_Posts")) \
    .orderBy(desc("Total_Posts")) \
    .limit(15) \
    .toPandas()

plt.figure(figsize=(10, 6))

plt.bar(
    sentiment_data["Sentiment"],
    sentiment_data["Total_Posts"]
)

plt.title("Top 15 Sentiment / Emotion Categories")
plt.xlabel("Sentiment / Emotion")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/sentiment_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 7. COUNTRY DISTRIBUTION
# ============================================================

print("Creating Country Distribution chart...")

country_data = df.groupBy("Country") \
    .agg(count("*").alias("Total_Posts")) \
    .orderBy(desc("Total_Posts")) \
    .limit(10) \
    .toPandas()

plt.figure(figsize=(10, 6))

plt.bar(
    country_data["Country"],
    country_data["Total_Posts"]
)

plt.title("Top 10 Countries by Social Media Posts")
plt.xlabel("Country")
plt.ylabel("Number of Posts")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/country_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. HOURLY ACTIVITY
# ============================================================

print("Creating Hourly Activity chart...")

hour_data = df.groupBy("Hour") \
    .agg(count("*").alias("Total_Posts")) \
    .orderBy("Hour") \
    .toPandas()

plt.figure(figsize=(10, 6))

plt.plot(
    hour_data["Hour"],
    hour_data["Total_Posts"],
    marker="o"
)

plt.title("Social Media Activity by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Posts")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/hourly_activity.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. PLATFORM ENGAGEMENT
# ============================================================

print("Creating Platform Engagement chart...")

engagement_data = df.groupBy("Platform") \
    .agg(
        avg("Likes").alias("Average_Likes"),
        avg("Retweets").alias("Average_Retweets")
    ) \
    .orderBy(desc("Average_Likes")) \
    .toPandas()

plt.figure(figsize=(8, 5))

x = range(len(engagement_data))

plt.bar(
    x,
    engagement_data["Average_Likes"],
    label="Average Likes"
)

plt.bar(
    x,
    engagement_data["Average_Retweets"],
    bottom=engagement_data["Average_Likes"],
    label="Average Retweets"
)

plt.xticks(
    x,
    engagement_data["Platform"]
)

plt.title("Average Engagement by Platform")
plt.xlabel("Platform")
plt.ylabel("Average Engagement")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/platform_engagement.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. TOP HASHTAGS
# ============================================================

print("Creating Top Hashtags chart...")

from pyspark.sql.functions import explode, split

hashtag_data = df \
    .filter(
        col("Hashtags").isNotNull() &
        (trim(col("Hashtags")) != "")
    ) \
    .withColumn(
        "Hashtag",
        explode(
            split(
                trim(col("Hashtags")),
                r"\s+"
            )
        )
    ) \
    .withColumn(
        "Hashtag",
        trim(col("Hashtag"))
    ) \
    .filter(col("Hashtag") != "") \
    .groupBy("Hashtag") \
    .agg(count("*").alias("Frequency")) \
    .orderBy(desc("Frequency")) \
    .limit(10) \
    .toPandas()

plt.figure(figsize=(10, 6))

plt.bar(
    hashtag_data["Hashtag"],
    hashtag_data["Frequency"]
)

plt.title("Top 10 Hashtags")
plt.xlabel("Hashtag")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/top_hashtags.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. ENGAGEMENT BY HOUR
# ============================================================

print("Creating Engagement by Hour chart...")

hour_engagement = df.groupBy("Hour") \
    .agg(
        avg("Likes").alias("Average_Likes"),
        avg("Retweets").alias("Average_Retweets")
    ) \
    .orderBy("Hour") \
    .toPandas()

plt.figure(figsize=(10, 6))

plt.plot(
    hour_engagement["Hour"],
    hour_engagement["Average_Likes"],
    marker="o",
    label="Average Likes"
)

plt.plot(
    hour_engagement["Hour"],
    hour_engagement["Average_Retweets"],
    marker="o",
    label="Average Retweets"
)

plt.title("Average Engagement by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Engagement")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{chart_folder}/engagement_by_hour.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FINISH
# ============================================================

print("\n========== VISUALIZATION COMPLETED ==========")

print("\nCharts saved in:")
print(chart_folder)

print("\nGenerated files:")
print("1. platform_distribution.png")
print("2. sentiment_distribution.png")
print("3. country_distribution.png")
print("4. hourly_activity.png")
print("5. platform_engagement.png")
print("6. top_hashtags.png")
print("7. engagement_by_hour.png")

spark.stop()