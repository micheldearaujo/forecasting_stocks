# Databricks notebook source
# MAGIC %md
# MAGIC ## Data cleaning
# MAGIC 
# MAGIC **Objective**: This notebook's objective is to load the raw dataset and perform cleaning on it.
# MAGIC 
# MAGIC **Takeaways**:
# MAGIC 
# MAGIC - Droped a categorical column to facilitate the mini project

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Read the fish data

# COMMAND ----------

fishDF = spark.sql("SELECT * FROM fish;")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Drop the categorical column

# COMMAND ----------

fishDF = fishDF.drop("Species")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Train test Split

# COMMAND ----------

train, test = fishDF.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Save the processed table

# COMMAND ----------

fishDF.write.mode("overwrite").saveAsTable("default.fish_cleaned")
train.write.mode("overwrite").saveAsTable("default.fish_cleaned_training")
test.write.mode("overwrite").saveAsTable("default.fish_cleaned_testing")

# COMMAND ----------


