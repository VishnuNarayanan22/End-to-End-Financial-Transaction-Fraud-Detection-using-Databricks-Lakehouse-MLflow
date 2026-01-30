# Catalog and schema setup
spark.sql("CREATE CATALOG IF NOT EXISTS fraud_project")
spark.sql("USE CATALOG fraud_project")
spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")

# Read raw CSV
raw_path = "/Volumes/fraud_project/raw/input_data/CreditCardData.csv"

bronze_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(raw_path)
)

from pyspark.sql.functions import current_timestamp

bronze_df = bronze_df.withColumn("ingestion_ts", current_timestamp())

# Clean column names
def clean_columns(df):
    for c in df.columns:
        df = df.withColumnRenamed(
            c, c.lower().replace(" ", "_").replace("-", "_")
        )
    return df

bronze_df_clean = clean_columns(bronze_df)

# Write Bronze Delta table
bronze_df_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("fraud_project.bronze.transactions")
