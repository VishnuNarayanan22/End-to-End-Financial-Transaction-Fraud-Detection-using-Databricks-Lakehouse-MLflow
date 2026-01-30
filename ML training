spark.sql("USE CATALOG fraud_project")
spark.sql("USE SCHEMA gold")

df_gold = spark.table("fraud_project.gold.fraud_features")

from pyspark.sql.functions import col

ml_df = df_gold.select(
    col("fraud").alias("label"),
    col("amount_gbp"),
    col("entry_mode"),
    col("type_of_card")
).dropna()

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

entry_indexer = StringIndexer(
    inputCol="entry_mode",
    outputCol="entry_mode_idx",
    handleInvalid="keep"
)

card_indexer = StringIndexer(
    inputCol="type_of_card",
    outputCol="type_of_card_idx",
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=["amount_gbp", "entry_mode_idx", "type_of_card_idx"],
    outputCol="features"
)

pipeline = Pipeline(stages=[entry_indexer, card_indexer, assembler])

final_ml_df = pipeline.fit(ml_df).transform(ml_df)

train_df, test_df = final_ml_df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow

evaluator = BinaryClassificationEvaluator(labelCol="label")

mlflow.set_experiment("/Shared/fraud_detection")

# Logistic Regression
lr = LogisticRegression(labelCol="label", featuresCol="features")
lr_model = lr.fit(train_df)
lr_auc = evaluator.evaluate(lr_model.transform(test_df))

# Random Forest
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50, maxDepth=8)
rf_model = rf.fit(train_df)
rf_auc = evaluator.evaluate(rf_model.transform(test_df))

# GBT (Champion)
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=30, maxDepth=5)
gbt_model = gbt.fit(train_df)
gbt_auc = evaluator.evaluate(gbt_model.transform(test_df))

with mlflow.start_run(run_name="GBT_Champion"):
    mlflow.log_param("model", "GBTClassifier")
    mlflow.log_metric("auc", gbt_auc)
    mlflow.spark.log_model(gbt_model, "fraud_gbt_model")
