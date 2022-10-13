# Databricks notebook source
# MAGIC %md
# MAGIC ## Data cleaning
# MAGIC 
# MAGIC **Objective**: This notebook's objective is to load the raw dataset and perform cleaning on it.
# MAGIC 
# MAGIC **Takeaways**:

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Read the raw data

# COMMAND ----------

temperatures_sdf = (spark
     .read
     .option('header', 'true')
      .option('inferSchema', 'true')
     .csv(f'/tmp/data/{RAW_DATASET_NAME}.csv')
)

# COMMAND ----------

display(temperatures_sdf)

# COMMAND ----------

temperatures_sdf.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Cleaning the data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Converting to monthly a average

# COMMAND ----------

temperatures_sdf = temperatures_sdf.withColumn("Year", F.year("Date"))
temperatures_sdf = temperatures_sdf.withColumn("Month", F.month("Date"))
temperatures_sdf = temperatures_sdf.withColumn("dateMonth", F.concat_ws('-', F.col("Year"), F.col("Month")))
temperatures_sdf = temperatures_sdf.withColumn("dateMonth", F.to_timestamp(F.col("dateMonth")))
temperatures_sdf = temperatures_sdf.groupBy("dateMonth").agg(F.avg("Temp").alias("avgTemp")).orderBy("dateMonth")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Save the cleaned data

# COMMAND ----------

temperatures_sdf.write.mode("overwrite").saveAsTable("temperatures_df")

# COMMAND ----------

(temperatures_sdf
 .write
 .options(header = 'True', delimiter=',')
 .mode("overwrite")
 .csv(f'/tmp/data/{INTERIM_DATASET_NAME}.csv')
)
