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
# MAGIC #### There is no cleaning to do

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Save the cleaned data

# COMMAND ----------

(temperatures_sdf
 .write
 .options(header = 'True', delimiter=',')
 .mode("overwrite")
 .csv(f'/tmp/data/{INTERIM_DATASET_NAME}.csv')
)
