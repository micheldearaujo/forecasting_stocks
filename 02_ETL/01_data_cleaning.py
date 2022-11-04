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

display(fishDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Drop the categorical column

# COMMAND ----------

fishDF = fishDF.drop("Species")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Save the processed table

# COMMAND ----------

fishDF.write.mode("overwrite").saveAsTable("default.fish_cleaned")

# COMMAND ----------


