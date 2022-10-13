# Databricks notebook source
# MAGIC %md
# MAGIC ## Econometric Modeling: Using AutoArima
# MAGIC 
# MAGIC **Objective**: This notebook's objective is
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
# MAGIC ### 4.0 Building the AutoArima

# COMMAND ----------



# COMMAND ----------

with mlflow.start_run() as run:
    model = auto_arima(train,
                      start_p = 0,
                      d = 1,
                      start_q = 0,
                      max_p = 12,
                      max_d = 5,
                      max_q = 12,
                      start_P = 0,
                      max_P = 12,
                      D=1,
                      start_Q = 0,
                      max_D = 5,
                      max_Q = 12,
                      m = 12,
                      seasonal = True,
                      error_action = 'warn',
                      trace=True,
                      supress_warnings=True,
                      stepwise=True,
                      random_state=42,
                      n_fits= 50)

# COMMAND ----------

model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Predictions

# COMMAND ----------

predictions = pd.DataFrame(model.predict(n_periods=forecast_horizon),
                          index=test.index)
predictions.columns = ['Predicted']
predictions.head()

# COMMAND ----------

model_r2 = r2_score(test['avgTemp'], predictions['Predicted'])
model_mape = np.sqrt(mean_absolute_percentage_error(test['avgTemp'], predictions['Predicted']))
model_rmse = np.sqrt(mean_squared_error(test['avgTemp'], predictions['Predicted']))
model_mae = mean_absolute_error(test['avgTemp'], predictions['Predicted'])
forecast_bias = ((predictions['Predicted'].sum() / test['avgTemp'].sum()) - 1)*100

print(f"Model R2: {round(model_r2, 2)}")
print(f"Model MAPE: {round(model_mape, 2)}")
print(f"Model RMSE: {round(model_rmse, 2)}")
print(f"Model MAE: {round(model_mae, 2)}")
print(f"Model MAE: {round(model_mae, 2)}")
print(f"Model Forecast Bias: {round(forecast_bias, 2)}%")

# COMMAND ----------

mlflow.log_metric("MAE", model_mae)

# COMMAND ----------

fig, axs = plt.subplots(figsize=(18, 8))
sns.lineplot(data=temp_df,
            x=temp_df.index,
            y='avgTemp', ax=axs,
            label='Training')

sns.lineplot(data=predictions,
            x=predictions.index,
            y='Predicted', ax=axs,
            label='Predictions')

axs.set_title(f"Baseline forecast - Seasonal Naïve\n RMSE = {round(model_rmse, 2)} | MAE = {round(model_mae, 2)} | MAPE = {round(model_mape, 2)} | FB = {round(forecast_bias, 2)}%")
axs.set_xlabel("Date")
axs.set_ylabel("Average Temperature")
plt.show()

# COMMAND ----------

model

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

with mlflow.start_run(run_name='SARIMA') as run:
    model = ARIMA(train, order=model.order,seasonal_order=model.seasonal_order)
    
    predictions = model.predict(n_periods=forecast_horizon, )
    predictions_df = pd.DataFrame(model.predict(n_periods=forecast_horizon),
                          index=test.index)
    predictions_df.columns = ['Predicted']
    
    model_r2 = r2_score(test['avgTemp'], predictions['Predicted'])
    model_mape = np.sqrt(mean_absolute_percentage_error(test['avgTemp'], predictions['Predicted']))
    model_rmse = np.sqrt(mean_squared_error(test['avgTemp'], predictions['Predicted']))
    model_mae = mean_absolute_error(test['avgTemp'], predictions['Predicted'])
    forecast_bias = ((predictions['Predicted'].sum() / test['avgTemp'].sum()) - 1)*100

    mlflow.log_metric("R2", round(model_r2, 2))
    mlflow.log_metric("MAPE", round(model_mape, 2))
    mlflow.log_metric("RMSE", round(model_rmse, 2))
    mlflow.log_metric("MAE", round(model_mae, 2))
    mlflow.log_metric("FB", round(forecast_bias, 2))
    

# COMMAND ----------


