from pyspark.sql.functions import col, when, isnan
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow

# Load Gold features
df = spark.table("fraud_project.gold.fraud_features")

# Handle nulls
df_clean = (
    df
    .withColumn(
        "amount_gbp",
        when(col("amount_gbp").isNull() | isnan("amount_gbp"), 0.0)
        .otherwise(col("amount_gbp"))
    )
    .withColumn(
        "is_foreign_transaction",
        when(col("is_foreign_transaction").isNull(), 0)
        .otherwise(col("is_foreign_transaction"))
    )
    .withColumn(
        "is_foreign_shipping",
        when(col("is_foreign_shipping").isNull(), 0)
        .otherwise(col("is_foreign_shipping"))
    )
    .withColumn(
        "is_night_transaction",
        when(col("is_night_transaction").isNull(), 0)
        .otherwise(col("is_night_transaction"))
    )
)

# Feature indexing
card_indexer = StringIndexer(
    inputCol="type_of_card",
    outputCol="type_of_card_idx",
    handleInvalid="keep"
)

entry_indexer = StringIndexer(
    inputCol="entry_mode",
    outputCol="entry_mode_idx",
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=[
        "amount_gbp",
        "is_foreign_transaction",
        "is_foreign_shipping",
        "is_night_transaction",
        "type_of_card_idx",
        "entry_mode_idx"
    ],
    outputCol="features",
    handleInvalid="keep"
)

# GBT model (Champion)
gbt = GBTClassifier(
    labelCol="fraud",
    featuresCol="features",
    maxIter=30,
    maxDepth=5,
    stepSize=0.1,
    seed=42
)

pipeline = Pipeline(stages=[
    card_indexer,
    entry_indexer,
    assembler,
    gbt
])

# Train-test split
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

# Train model
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

# Evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="fraud",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"GBT AUC: {auc}")

# MLflow logging
mlflow.set_experiment("/Shared/fraud_detection")

with mlflow.start_run(run_name="GBT_Champion_Model"):
    mlflow.log_param("model", "GradientBoostedTrees")
    mlflow.log_param("maxIter", 30)
    mlflow.log_param("maxDepth", 5)
    mlflow.log_metric("auc", auc)
    mlflow.spark.log_model(model, "fraud_gbt_model")
