# Databricks notebook source
# MAGIC %md
# MAGIC ## Time series distribution EDA
# MAGIC 
# MAGIC **Objectives**: Analyse the Time Series characteristics in order to check if the data is good to build a *Forecasting* model.
# MAGIC 
# MAGIC - Normality of distribution
# MAGIC - Stationarity
# MAGIC - Autocorrelation
# MAGIC - Trends
# MAGIC - Seasonality
# MAGIC - Missing values
# MAGIC 
# MAGIC **Takeaways**: 
# MAGIC 
# MAGIC - The distribution is normal;
# MAGIC - There are no missing values;
# MAGIC - Only 0.05% of zero values;
# MAGIC - The series is Stationary;
# MAGIC - 

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
     .csv(f'/tmp/data/{RAW_DATASET_NAME}.csv')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Basic data summary

# COMMAND ----------

display(temperatures_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC So we have basically no missing values, only 0.05% of zero values and a normal distribution!

# COMMAND ----------

temp_pd = temperatures_sdf.toPandas()
temp_pd = temp_pd.set_index("Date")
temp_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Stationarity test

# COMMAND ----------

test_result = adfuller(x=temp_pd, maxlag=12)
print(f"statistic: {test_result[0]}")
print(f"p-value: {test_result[1]}")
print(f"number of lags: {test_result[2]}")

# COMMAND ----------

# MAGIC %md
# MAGIC The series is Stationary!

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Autocorrelation Analysis

# COMMAND ----------

# Plot both autocorrelation and partial autocorrelation
fig, axs = plt.subplots(1, 2, figisze=(20, 12))
plot_acf(x=temp_pd, ax= axs[0])


# COMMAND ----------


