from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum,
    trim,
    lower,
    regexp_replace,
    to_date
)

spark = SparkSession.builder \
    .appName("SparkPulse AI - Data Preprocessing") \
    .master("local[*]") \
    .getOrCreate()

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = spark.read.csv(
    "data/social_media.csv",
    header=True,
    inferSchema=True
)

print("\n========== ORIGINAL DATASET ==========")
print("Number of Rows:", df.count())

# ============================================================
# 2. REMOVE EXTRA COLUMNS
# ============================================================

df = df.drop("_c0", "Unnamed: 0")

print("\n========== AFTER REMOVING EXTRA COLUMNS ==========")
print("Number of Rows:", df.count())

# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.select([
    sum(col(column).isNull().cast("int")).alias(column)
    for column in df.columns
])

missing_values.show()

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

print("\n========== DUPLICATE CHECK ==========")

rows_before = df.count()

df = df.dropDuplicates()

rows_after = df.count()

duplicates_removed = rows_before - rows_after

print("Rows Before Removing Duplicates:", rows_before)
print("Rows After Removing Duplicates:", rows_after)
print("Duplicate Rows Removed:", duplicates_removed)

# ============================================================
# 5. CLEAN TEXT
# ============================================================

print("\n========== TEXT CLEANING ==========")

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

print("Text cleaning completed successfully.")

# ============================================================
# 6. CLEAN SENTIMENT LABELS
# ============================================================

print("\n========== SENTIMENT LABEL CLEANING ==========")

df = df.withColumn(
    "Sentiment",
    trim(col("Sentiment"))
)

print("Sentiment labels cleaned successfully.")

# ============================================================
# 7. DATE PROCESSING
# ============================================================

print("\n========== DATE PROCESSING ==========")

df = df.withColumn(
    "Date",
    to_date(col("Timestamp"))
)

print("Date column created successfully.")

print("\nDate and Time Sample:")

df.select(
    "Timestamp",
    "Date",
    "Year",
    "Month",
    "Day",
    "Hour"
).show(5)

# ============================================================
# 8. SENTIMENT CATEGORIES
# ============================================================

print("\n========== SENTIMENT CATEGORIES ==========")

df.groupBy("Sentiment") \
    .count() \
    .orderBy(col("count").desc()) \
    .show(100, truncate=False)

# ============================================================
# 9. FINAL DATASET INFORMATION
# ============================================================

print("\n========== FINAL PREPROCESSED DATASET ==========")

print("Final Number of Rows:", df.count())

print("\nFinal Columns:")
print(df.columns)

print("\nFinal Schema:")
df.printSchema()

# ============================================================
# STOP SPARK
# ============================================================

spark.stop()