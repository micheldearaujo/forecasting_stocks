# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Ingestion
# MAGIC 
# MAGIC **Objective**: Download or upload the csv files from the web and create PySpark DataFrames or Delta Tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Download the Minimum Temperatures dataset

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Download the dataset from the Web

# COMMAND ----------

spark.sparkContext.addFile(DATASET_URL)

temperatures_sdf = (spark
      .read
      .csv("file://"+SparkFiles.get(URL_DATASET_NAME), header=True, inferSchema= True)
 )

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Create a folder within the DBFS

# COMMAND ----------

if os.path.exists(DATA_PATH):
    print("Path Already Exists")
else:
    print("Creating Data Path")
    os.mkdir(DATA_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Save the file as CSV in the DBFS

# COMMAND ----------

(temperatures_sdf
 .write
 .options(header = 'True', delimiter=',')
 .mode("overwrite")
 .csv(f'/tmp/data/{RAW_DATASET_NAME}.csv')
)
