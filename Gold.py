spark.sql("USE CATALOG fraud_project")
spark.sql("USE SCHEMA silver")

silver_df = spark.table("fraud_project.silver.transactions")

from pyspark.sql import functions as F

gold_features_df = (
    silver_df
    .withColumn(
        "is_foreign_transaction",
        F.when(
            F.col("country_of_transaction") != F.col("country_of_residence"), 1
        ).otherwise(0)
    )
    .withColumn(
        "is_foreign_shipping",
        F.when(
            F.col("shipping_address") != F.col("country_of_residence"), 1
        ).otherwise(0)
    )
    .withColumn(
        "is_night_transaction",
        F.when(
            (F.col("time") >= 22) | (F.col("time") <= 5), 1
        ).otherwise(0)
    )
    .select(
        "transaction_id",
        "amount_gbp",
        "is_foreign_transaction",
        "is_foreign_shipping",
        "is_night_transaction",
        "type_of_card",
        "entry_mode",
        "fraud",
        "transaction_date"
    )
)

spark.sql("USE SCHEMA gold")

gold_features_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("fraud_project.gold.fraud_features")

spark.sql("OPTIMIZE fraud_project.gold.fraud_features")
