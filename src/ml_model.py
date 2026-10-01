import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, regexp_replace, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ============================================================
# 1. START SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("SparkPulse AI - Sentiment ML") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n========== SPARK ML SENTIMENT MODEL ==========")

# ============================================================
# 2. LOAD DATA
# ============================================================

df = spark.read.csv(
    "data/social_media.csv",
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

df = df.withColumn(
    "Sentiment",
    trim(col("Sentiment"))
)

# ============================================================
# 4. CONVERT 191 EMOTIONS INTO 3 SENTIMENT CLASSES
# ============================================================

positive_words = [
    "positive", "joy", "happy", "happiness", "joyfulreunion",
    "excitement", "enthusiasm", "euphoria", "elation", "ecstasy",
    "gratitude", "grateful", "appreciation", "hope", "hopeful",
    "optimism", "positivity", "love", "affection", "adoration",
    "admiration", "inspiration", "inspired", "empowerment",
    "confidence", "confident", "pride", "proud", "success",
    "triumph", "celebration", "enjoyment", "amusement",
    "satisfaction", "fulfillment", "accomplishment", "resilience",
    "relief", "rejuvenation", "harmony", "serenity", "calmness",
    "tranquility", "wonder", "wonderment", "awe", "amazement",
    "captivation", "enchantment", "thrill", "vibrancy", "zest",
    "playful", "playfuljoy", "kind", "kindness", "compassion",
    "compassionate", "empathetic", "sympathy", "tenderness",
    "heartwarming", "friendship", "connection", "adventure",
    "exploration", "motivation", "breakthrough", "blessed",
    "freedom", "festivejoy", "radiance", "renewedeffort"
]

negative_words = [
    "negative", "sad", "sadness", "sorrow", "grief", "despair",
    "desolation", "devastated", "disappointed", "disappointment",
    "anger", "hate", "frustration", "frustrated", "resentment",
    "bitterness", "bitter", "betrayal", "heartbreak", "heartache",
    "helplessness", "loneliness", "isolation", "loss", "suffering",
    "fear", "fearful", "anxiety", "apprehensive", "embarrassed",
    "shame", "regret", "numbness", "overwhelmed", "exhaustion",
    "boredom", "indifference", "dismissive", "envious", "envy",
    "jealous", "jealousy", "pressure", "darkness", "desperation",
    "miscalculation", "ruins", "yearning"
]

from pyspark.sql.functions import when

df = df.withColumn(
    "Label",
    when(
        lower(col("Sentiment")).isin(positive_words),
        "Positive"
    ).when(
        lower(col("Sentiment")).isin(negative_words),
        "Negative"
    ).otherwise(
        "Neutral"
    )
)

print("\n========== SENTIMENT CLASS DISTRIBUTION ==========")

df.groupBy("Label") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()

# ============================================================
# 5. REMOVE EMPTY TEXT
# ============================================================

df = df.filter(
    col("Text").isNotNull() &
    (col("Text") != "")
)

# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

train_data, test_data = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training Rows:", train_data.count())
print("Testing Rows:", test_data.count())

# ============================================================
# 7. TOKENIZATION
# ============================================================

tokenizer = Tokenizer(
    inputCol="Text",
    outputCol="Words"
)

# ============================================================
# 8. TF-IDF
# ============================================================

hashingTF = HashingTF(
    inputCol="Words",
    outputCol="RawFeatures",
    numFeatures=1000
)

idf = IDF(
    inputCol="RawFeatures",
    outputCol="Features"
)

# ============================================================
# 9. CONVERT LABEL TO NUMERIC
# ============================================================

label_indexer = StringIndexer(
    inputCol="Label",
    outputCol="LabelIndex"
)

# ============================================================
# 10. LOGISTIC REGRESSION
# ============================================================

lr = LogisticRegression(
    featuresCol="Features",
    labelCol="LabelIndex",
    maxIter=50
)

# ============================================================
# 11. CREATE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    stages=[
        tokenizer,
        hashingTF,
        idf,
        label_indexer,
        lr
    ]
)

# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n========== TRAINING MODEL ==========")

model = pipeline.fit(train_data)

print("Model Training Completed!")

# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

predictions = model.transform(test_data)

print("\n========== SAMPLE PREDICTIONS ==========")

predictions.select(
    "Text",
    "Label",
    "prediction"
).show(10, truncate=False)

# ============================================================
# 14. ACCURACY
# ============================================================

evaluator = MulticlassClassificationEvaluator(
    labelCol="LabelIndex",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n========== CONFUSION MATRIX ==========")

confusion_matrix = predictions.groupBy(
    "LabelIndex",
    "prediction"
).count().orderBy(
    "LabelIndex",
    "prediction"
)

confusion_matrix.show(20)

# ============================================================
# 16. TEST PREDICTION
# ============================================================

print("\n========== TEST PREDICTION ==========")

sample_text = spark.createDataFrame([
    ("I am very happy and excited about this amazing day!",)
], ["Text"])

# Add a temporary neutral label so the pipeline can transform
sample_text = sample_text.withColumn(
    "Label",
    lit("Neutral")
)

sample_prediction = model.transform(sample_text)

sample_prediction.select(
    "Text",
    "prediction"
).show(truncate=False)

# ============================================================
# 17. STOP SPARK
# ============================================================

spark.stop()

print("\n========== SPARK ML COMPLETED ==========")