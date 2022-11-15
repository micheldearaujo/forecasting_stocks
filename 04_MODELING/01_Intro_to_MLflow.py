# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/concepts.html" target="_blank">MLflow</a> seeks to address these three core issues:
# MAGIC 
# MAGIC * It’s difficult to keep track of experiments
# MAGIC * It’s difficult to reproduce code
# MAGIC * There’s no standard way to package and deploy models
# MAGIC 
# MAGIC In the past, when examining a problem, you would have to manually keep track of the many models you created, as well as their associated parameters and metrics. This can quickly become tedious and take up valuable time, which is where MLflow comes in.
# MAGIC 
# MAGIC MLflow is pre-installed on the Databricks Runtime for ML.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC * Use MLflow to track experiments, log metrics, and compare runs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# Create a name for this run
RUN_NAME = ...

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Load the data

# COMMAND ----------

# Load the data and split between X and y
train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()

X = ...
y = ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Train a regressor model without hyperparameter tuning
# MAGIC Let's take a step back and remember the usual way we train our models

# COMMAND ----------

# Create and instance
model = <...>

# Fit the model
model.fit(...)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Validate

# COMMAND ----------

# Perform predictions


# Get an evaluation metric
mape = ...
r2 = ...

# COMMAND ----------

# Plot the scatter plot True versus Predicted

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

# Define some parameters
max_depth = 5
min_samples_leaf = 3
min_samples_split = 3
n_estimators = 200
random_state = 42

# COMMAND ----------

# Create and instance
model = RandomForestRegressor(
...
)

# Fit the model
...

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 Validate

# COMMAND ----------

# Perform predictions
y_pred = 

# Get an evaluation metric
mape = ...
r2 = ...

# COMMAND ----------

# Plot the scatter plot True versus Predicted

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Using MLflow
# MAGIC <i18n value="9ab8c080-9012-4f38-8b01-3846c1531a80"/>
# MAGIC 
# MAGIC #### MLflow Tracking
# MAGIC 
# MAGIC MLflow Tracking is a logging API specific for machine learning and agnostic to libraries and environments that do the training.  It is organized around the concept of **runs**, which are executions of data science code.  Runs are aggregated into **experiments** where many runs can be a part of a given experiment and an MLflow server can host many experiments.
# MAGIC 
# MAGIC You can use <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a> to set an experiment, but if you do not specify an experiment, it will automatically be scoped to this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Just run it simple

# COMMAND ----------

# MAGIC %md <i18n value="82786653-4926-4790-b867-c8ccb208b451"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Track Runs
# MAGIC 
# MAGIC Each run can record the following information:<br><br>
# MAGIC 
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC 
# MAGIC **NOTE**: For Spark models, MLflow can only log PipelineModels.

# COMMAND ----------

# Run it with MLFlow
#with mlflow.start_run() ...

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Try to log some metrics and parameters

# COMMAND ----------

max_depth = 5
min_samples_leaf = 3
min_samples_split = 3
n_estimators = 200
random_state = 42

# COMMAND ----------

# Run the MLflow with logging metrics, parameters and artifacts

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Query and compare past runs
# MAGIC 
# MAGIC You can query past runs programmatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.

# COMMAND ----------

from mlflow...

# COMMAND ----------

# You can list the experiments using the search_experiments() method.
# It will return all the experiment available withing the Server
client = ...

# COMMAND ----------

# Query past runs within this notebook
runs = ...
runs.head()

# COMMAND ----------

# Query past runs with the Experiment ID
experiment_id = run...

# COMMAND ----------

runs = mlflow.search_runs(
    
    # We can order by some column
                         )

# COMMAND ----------

# MAGIC %md
# MAGIC #### That's all for today!

# COMMAND ----------


