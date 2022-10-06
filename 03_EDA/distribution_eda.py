# Databricks notebook source
# MAGIC %md
# MAGIC ## Time series distribution EDA
# MAGIC 
# MAGIC **Objectives**: Analyse the Time Series characteristics in order to check if the data is good to build a *Forecasting* model.
# MAGIC 
# MAGIC - Normality of distribution
# MAGIC - Stationarity
# MAGIC - Granger Causality
# MAGIC - Autocorrelation
# MAGIC - Trends
# MAGIC - Seasonality
# MAGIC - Missing values

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Reading

# COMMAND ----------

temperatures_sdf = (spark
     .read
     .option('header', 'true')
      .option('inferSchema', 'true')
     .csv(f'/tmp/data/{INTERIM_DATASET_NAME}.csv')
)

# COMMAND ----------


