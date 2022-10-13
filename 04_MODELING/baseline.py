# Databricks notebook source
# MAGIC %md
# MAGIC ## Econometric Modeling: Seasonal Naïve baseline
# MAGIC 
# MAGIC **Objective**: This notebook's objective is to create a Seasonal Naïve model to serve as Baseline for our models benchmark
# MAGIC 
# MAGIC **Takeaways**: The key takeaways of this notebook are:
# MAGIC 
# MAGIC -
# MAGIC -
# MAGIC -
# MAGIC -
# MAGIC -

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

temperatures_sdf = (spark
     .read
     .option('header', 'true')
      .option('inferSchema', 'true')
     .csv(f'/tmp/data/{INTERIM_DATASET_NAME}.csv')                   
).toPandas()

temp_df = temperatures_sdf.set_index("dateMonth")

# COMMAND ----------

temp_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Train and test split

# COMMAND ----------

train = temp_df[:temp_df.shape[0] - forecast_horizon]
test = temp_df[temp_df.shape[0] - forecast_horizon:]

# COMMAND ----------

fig, axs = plt.subplots(figsize=(18, 8))
sns.lineplot(data=train,
            x=train.index,
            y='avgTemp', ax=axs,
            label='Training')
sns.lineplot(data=test,
            x=test.index,
            y='avgTemp', ax=axs,
            label='Testing')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Building the SNaïve

# COMMAND ----------

def pysnaive(train_series,seasonal_periods,forecast_horizon):
    '''
    Python implementation of Seasonal Naive Forecast. 
    This should work similar to https://otexts.com/fpp2/simple-methods.html
    Returns two arrays
     > fitted: Values fitted to the training dataset
     > fcast: seasonal naive forecast
    
    Author: Sandeep Pawar
    
    Date: Apr 9, 2020
    
    Ver: 1.0
    
    train_series: Pandas Series
        Training Series to be used for forecasting. This should be a valid Pandas Series. 
        Length of the Training set should be greater than or equal to number of seasonal periods
        
    Seasonal_periods: int
        No of seasonal periods
        Yearly=1
        Quarterly=4
        Monthly=12
        Weekly=52
        

    Forecast_horizon: int
        Number of values to forecast into the future
    
    e.g. 
    fitted_values = pysnaive(train,12,12)[0]
    fcast_values = pysnaive(train,12,12)[1]
    '''
    
    if len(train_series)>= seasonal_periods: #checking if there are enough observations in the training data
        
        last_season=train_series.iloc[-seasonal_periods:]
        
        reps=np.int(np.ceil(forecast_horizon/seasonal_periods))
        
        fcarray=np.tile(last_season,reps)
        
        fcast=pd.Series(fcarray[:forecast_horizon])
        
        fitted = train_series.shift(seasonal_periods)
        
    else:
        fcast=print("Length of the trainining set must be greater than number of seasonal periods") 
    
    return fitted, fcast

# COMMAND ----------

predictions = test.copy()

# COMMAND ----------

#Fitted values
py_snaive_fit = pysnaive(train["avgTemp"], 
                     seasonal_periods=12,
                     forecast_horizon=18)[0]

#forecast
py_snaive = pysnaive(train["avgTemp"], 
                     seasonal_periods=12,
                     forecast_horizon=18)[1]

#Residuals
py_snaive_resid = (train["avgTemp"] - py_snaive_fit).dropna()




predictions["py_snaive"] = py_snaive.values 


predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Evaluating

# COMMAND ----------

model_r2 = r2_score(predictions['avgTemp'], predictions['py_snaive'])
model_mape = np.sqrt(mean_absolute_percentage_error(predictions['avgTemp'], predictions['py_snaive']))
model_rmse = np.sqrt(mean_squared_error(predictions['avgTemp'], predictions['py_snaive']))
model_mae = mean_absolute_error(predictions['avgTemp'], predictions['py_snaive'])
forecast_bias = ((predictions['py_snaive'].sum() / predictions['avgTemp'].sum()) - 1)*100
print(f"Model R2: {round(model_r2, 2)}")
print(f"Model MAPE: {round(model_mape, 2)}")
print(f"Model RMSE: {round(model_rmse, 2)}")
print(f"Model MAE: {round(model_mae, 2)}")
print(f"Model Forecast Bias: {round(forecast_bias, 2)}%")

# COMMAND ----------

fig, axs = plt.subplots(figsize=(18, 8))
sns.lineplot(data=temp_df,
            x=temp_df.index,
            y='avgTemp', ax=axs,
            label='True Values')
#sns.lineplot(data=test,
#            x=test.index,
#            y='avgTemp', ax=axs,
#            label='Testing')

sns.lineplot(data=predictions,
            x=predictions.index,
            y='py_snaive', ax=axs,
            label='Seasonal Naïve Forecast')

axs.set_title(f"Baseline forecast - Seasonal Naïve\n RMSE = {round(model_rmse, 2)} | MAE = {round(model_mae, 2)} | MAPE = {round(model_mape, 2)} | FB = {round(forecast_bias, 2)}%")
axs.set_xlabel("Date")
axs.set_ylabel("Average Temperature")
plt.show()

# COMMAND ----------


