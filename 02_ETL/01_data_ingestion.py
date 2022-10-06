# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Ingestion
# MAGIC 
# MAGIC **Objective**: Download or upload the csv files from the web and create PySpark DataFrames or Delta Tables.

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

spark.sparkContext.addFile(DATASET_URL)

temperatures_sdf = (spark
      .read
      .csv("file://"+SparkFiles.get(URL_DATASET_NAME), header=True, inferSchema= True)
 )

# COMMAND ----------

if os.path.exists(DATA_PATH):
    print("Path Already Exists")
else:
    print("Creating Data Path")
    os.mkdir(DATA_PATH)

# COMMAND ----------

(temperatures_sdf
 .write
 .options(header = 'True', delimiter=',')
 .mode("overwrite")
 .csv(f'/tmp/data/{RAW_DATASET_NAME}.csv')
)
