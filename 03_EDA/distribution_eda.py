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
# MAGIC - There is a non constant trend
# MAGIC - There is yearly seasonality (12 months or 365 days)
# MAGIC 
# MAGIC We can choose to proceed with a daily model or a monthly model. We can test both granularity and compare the results.

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

temperatures_sdf = temperatures_sdf.withColumn("Year", F.year("Date"))
temperatures_sdf = temperatures_sdf.withColumn("Month", F.month("Date"))
temperatures_sdf = temperatures_sdf.withColumn("dateMonth", F.concat_ws('-', F.col("Year"), F.col("Month")))
temperatures_sdf = temperatures_sdf.withColumn("dateMonth", F.to_timestamp(F.col("dateMonth")))

# COMMAND ----------

temperatures_sdf_monthly = temperatures_sdf.groupBy("dateMonth").agg(F.avg("Temp").alias("avgTemp"))

# COMMAND ----------

display(temperatures_sdf_monthly)

# COMMAND ----------

# MAGIC %md
# MAGIC So we have basically no missing values, only 0.05% of zero values and a normal distribution!

# COMMAND ----------

temp_pd = temperatures_sdf.toPandas()
temp_pd = temp_pd.set_index("Date")
temp_pd.head()

# COMMAND ----------

temp_monthly_pd = temperatures_sdf_monthly.orderBy("dateMonth").toPandas()
temp_monthly_pd = temp_monthly_pd.set_index("dateMonth")
temp_monthly_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Stationarity test

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 On the daily level

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
# MAGIC #### 4.2 On the Monhtly level

# COMMAND ----------

test_result = adfuller(x=temp_monthly_pd, maxlag=12)
print(f"statistic: {test_result[0]}")
print(f"p-value: {test_result[1]}")
print(f"number of lags: {test_result[2]}")

# COMMAND ----------

# MAGIC %md
# MAGIC The series is NOT Stationary!

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Autocorrelation Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1 On the daily level

# COMMAND ----------

# Plot both autocorrelation and partial autocorrelation
fig, axs = plt.subplots(1, 2, figsize=(25, 8))
# Plot the autocorrelation function
plot_acf(x=temp_pd, ax= axs[0])
# Plot the partial autocorrelation function
plot_pacf(x=temp_pd, ax=axs[1])
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC It seems that the series is strongly autocorrelated, but when we analyse the partial autocorrelation, it is more autocorrelated only with the first lag and the third. We expected that we would have a yearly seasonality, but the data is daily, so we would have to watch until the 365 lag to see that.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 On the monthly level

# COMMAND ----------

# Plot both autocorrelation and partial autocorrelation
fig, axs = plt.subplots(1, 2, figsize=(25, 8))
# Plot the autocorrelation function
plot_acf(x=temp_monthly_pd, ax= axs[0])
# Plot the partial autocorrelation function
plot_pacf(x=temp_monthly_pd, ax=axs[1])
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC As we expected, there is a strong autocorrelation with the lag 12, as the whether is yearly seasonal.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.0 Seasonal decomposition
# MAGIC Now let's take a look at how the trends and the seasonality behaves. Maybe we can decide to change the granularity from daily to monthly based on that.

# COMMAND ----------

def plot_seasonal_decomposition(series, period):
    # Extract the seasonality data
    decomposition = seasonal_decompose(x=series,
                       period=period)
    
    # Plot the results:
    fig, axs = plt.subplots(4, 1, figsize=(20, 15))

    axs[0].plot(series)
    axs[0].set_title("Original")
    axs[1].plot(decomposition.trend)
    axs[1].set_title("Trend")
    axs[2].plot(decomposition.seasonal)
    axs[2].set_title("Seasonal")
    axs[3].plot(decomposition.resid)
    axs[3].set_title("Residual")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.1 Daily Level

# COMMAND ----------

plot_seasonal_decomposition(temp_pd, 365)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2 Monthly level

# COMMAND ----------

plot_seasonal_decomposition(temp_monthly_pd, 12)

# COMMAND ----------

# MAGIC %md
# MAGIC We Can see that nothing has chanced too much, the monthly data is just a smooth version of the daily. But we need to remember that those are average values, not true values.

# COMMAND ----------

temp_monthly_pd
