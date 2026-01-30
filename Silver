spark.sql("USE CATALOG fraud_project")
spark.sql("USE SCHEMA bronze")

bronze_df = spark.table("fraud_project.bronze.transactions")

from pyspark.sql import functions as F

silver_df = (
    bronze_df
    .withColumn("amount_gbp", F.regexp_replace("amount", "[£,]", "").cast("double"))
    .withColumn("transaction_date", F.to_date("date", "dd-MMM-yy"))
    .withColumn("fraud", F.col("fraud").cast("int"))
    .withColumn("processed_ts", F.current_timestamp())
)

# Business rule: high-risk transaction
silver_df = silver_df.withColumn(
    "is_high_risk",
    F.when(
        (F.col("amount_gbp") > 200) |
        (F.col("type_of_transaction") == "Online") |
        (F.col("country_of_transaction") != F.col("country_of_residence")),
        1
    ).otherwise(0)
)

silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("fraud_project.silver.transactions")

spark.sql("""
OPTIMIZE fraud_project.silver.transactions
ZORDER BY (country_of_transaction, fraud)
""")
