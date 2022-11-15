# Databricks notebook source
# MAGIC %md
# MAGIC ## Modeling a simple Random Forest using MLflow
# MAGIC 
# MAGIC O objetivo desta primeira seção é se familarizar com o MLflow e conhecer os conceitos básicos, como:
# MAGIC 
# MAGIC - Experiments
# MAGIC - Runs
# MAGIC - Tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

RUN_NAME = 'RandomForest_Hyperopt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Load the data

# COMMAND ----------

train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()
X = train_df.drop(model_config['TARGET_VARIABLE'], axis=1)
y = train_df[model_config['TARGET_VARIABLE']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Train a regressor model without hyperparameter tuning

# COMMAND ----------

# Create and instance
model = RandomForestRegressor()

# Fit the model
model.fit(
X_train,
y_train
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Validate

# COMMAND ----------

# Perform predictions
y_pred = model.predict(X_test)

# Get an evaluation metric
mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
r2 = round(r2_score(y_test, y_pred), 3)

# COMMAND ----------

fig, axs = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

# Create and instance
model = RandomForestRegressor(
    max_depth = 5,
    min_samples_leaf = 3,
    min_samples_split = 3,
    n_estimators = 200,
    random_state = 42,
)

# Fit the model
model.fit(
X_train,
y_train
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 Validate

# COMMAND ----------

# Perform predictions
y_pred = model.predict(X_test)

# Get an evaluation metric
mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
r2 = round(r2_score(y_test, y_pred), 3)

# COMMAND ----------

fig, axs = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
plt.title(f"Predicted versus Ground truth - With some optimal parameters\nR2 = {r2} | MAPE = {mape}")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Using MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Just run it simple

# COMMAND ----------

with mlflow.start_run(run_name = RUN_NAME) as run:
    # Create and instance
    model = RandomForestRegressor(
        max_depth = 5,
        min_samples_leaf = 3,
        min_samples_split = 3,
        n_estimators = 200,
        random_state = 42,
    )

    # Fit the model
    model.fit(
    X_train,
    y_train
    )
    
    ## Validate the model
    # Perform predictions
    y_pred = model.predict(X_test)

    # Get an evaluation metric
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    

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

with mlflow.start_run(run_name = RUN_NAME+'_with_log') as run:
    # Create and instance
    model = RandomForestRegressor(
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split,
        n_estimators = n_estimators,
        random_state = random_state,
    )

    # Fit the model
    model.fit(
    X_train,
    y_train
    )
    
    ## Validate the model
    # Perform predictions
    y_pred = model.predict(X_test)

    # Get an evaluation metric
    mape = round(mean_absolute_percentage_error(y_test, y_pred),3 )
    r2 = round(r2_score(y_test, y_pred), 3)
    
    
    # Log the metrics
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("R2", r2)
    
    # Log the parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    
    # Log the model
    mlflow.sklearn.log_model(model, "artifacts/random_forest_model")
    
    # Log the evaluation figure
    fig, axs = plt.subplots(figsize=(12, 8))

    plt.scatter(y_test, y_pred)
    plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    
    plt.savefig("artifacts/r2_figure.png")
    
    mlflow.log_artifact('artifacts/r2_figure.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Query past runs

# COMMAND ----------

from mlflow.tracking import MlflowClient

# COMMAND ----------

# Create a Client
client = MlflowClient()

# COMMAND ----------

# You can list the experiments using the list_experiments() method.
# It will return all the experiment available withing the Server
client.search_experiments()

# COMMAND ----------

# To query past runs within this experiment (this notebook)
runs = mlflow.search_runs()

# COMMAND ----------

runs.head()

# COMMAND ----------

# The search_runs function has tons of parameters. To search the experiments in another notebook, you have to get the Experiment ID.
# Later we will see how to get experiments from another notebook
# We can use the last run to get the experiment id
experiment_id = run.info.experiment_id

# COMMAND ----------

runs = mlflow.search_runs([experiment_id],
                         order_by =["metrics.MAPE ASC"] # We can order by some column
                         )

# COMMAND ----------

runs

# COMMAND ----------

# MAGIC %md
# MAGIC #### That's all for today!!

# COMMAND ----------


