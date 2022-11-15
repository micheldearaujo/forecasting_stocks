# Databricks notebook source
# MAGIC %md
# MAGIC ## Model training and selection pipeline
# MAGIC 
# MAGIC **Objective**: This notebook's objective run the model training and selection pipeline
# MAGIC 
# MAGIC **Takeaways**: The key takeaways of this notebook are:

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

start_time = time.monotonic()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Load the data

# COMMAND ----------

train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Split the dataset

# COMMAND ----------

X = train_df.drop(TARGET_VARIABLE, axis=1)
y = train_df[TARGET_VARIABLE]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Train the models

# COMMAND ----------

# MAGIC %run ../02_ETL/01_data_cleaning

# COMMAND ----------

# MAGIC %run ./extra_trees_hyperopt

# COMMAND ----------

# MAGIC %run ./xgboost_hyperopt

# COMMAND ----------

# MAGIC %run ./random_forest_hyperopt

# COMMAND ----------

# MAGIC %run ./lightgbm_hyperopt

# COMMAND ----------

end_time = time.monotonic()
processing_time = dt.timedelta(seconds = end_time - start_time)

print(f"The total processing time was: {processing_time}")

# COMMAND ----------

# max_evals = 10:
# :03:40.299664
# max_evals = 50:
# 0:12:10.928597
