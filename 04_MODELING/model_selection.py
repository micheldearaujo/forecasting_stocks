# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Selection
# MAGIC 
# MAGIC **Objective**: This notebook's objective is to perform the model selection and choose the new Staging model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

df = spark.sql("SELECT * FROM default.fish_cleaned").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Load the past experiment runs

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 define the paths

# COMMAND ----------

experiments_notebook = mlflow.get_experiment_by_name('/Repos/michel.araujo@artefact.com/time_series_mlops/04_MODELING/train')
experiments_ids = experiments_notebook.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Load the runs

# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()
runs = mlflow.search_runs(
  [experiments_ids],
  order_by=[f"metrics.{model_config['COMPARISON_METRIC']} ASC"]
)

# Filter the runs to only include the finished ones and the runs of the day
runs = runs[(runs['status'] == 'FINISHED')&\
             (runs['end_time'].dt.strftime('%Y-%m-%d') == dt.datetime.today().strftime('%Y-%m-%d'))]

# COMMAND ----------

runs

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Get the best run

# COMMAND ----------

# Get the best run
best_run = runs.iloc[0]
# Get the best run ID
best_run_id = best_run.run_id
# Get the best run name
best_model_name = best_run['tags.mlflow.runName']

# COMMAND ----------

best_model_name

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 Load this Model

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri = f"runs:/{best_run_id}/{best_model_name}")

# COMMAND ----------

model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.5 Register the model

# COMMAND ----------

# Register model
result = mlflow.register_model(
    model_uri = f"runs:/{best_run_id}/{best_model_name}",
    name = model_config['REGISTER_MODEL_NAME']
)

# COMMAND ----------

# Add a description to the model
client.update_model_version(
    name=model_config['REGISTER_MODEL_NAME'],
    version=result.version,
    description="ATF Meetup Model Info -> Model Type = {}".format(model)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Compare the new best run with the last Production/Staging model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Load the Staging model

# COMMAND ----------

# Getting the last model versions
models_versions = []
for mv in client.search_model_versions("name='{}'".format(model_config['REGISTER_MODEL_NAME'])):
    models_versions.append(dict(mv))

# COMMAND ----------

try: 
    # Get all the models that are in stage (In the majority of the cases it should be only one)
    current_model = [x for x in models_versions if x['current_stage'] == 'Staging'][0]
    # Extract the current staging model MAPE
    current_model_mape = mlflow.get_run(current_model['run_id']).data.metrics[model_config['COMPARISON_METRIC']]
    # Get the new model MAPE
    candidate_model_mape = mlflow.get_run(result.run_id).data.metrics[model_config['COMPARISON_METRIC']]
    
except:
    # If we get an error then we first set the first model to staging
    print("No Staging model founded. Register the candidate as staging.")
    
    client.transition_model_version_stage(
    name=model_config['REGISTER_MODEL_NAME'],
    version=result.version,
    stage='Staging',
    )

# COMMAND ----------

# Getting the last model versions
models_versions = []
for mv in client.search_model_versions("name='{}'".format(model_config['REGISTER_MODEL_NAME'])):
    models_versions.append(dict(mv))

# Get all the models that are in stage (In the majority of the cases it should be only one)
current_model = [x for x in models_versions if x['current_stage'] == 'Staging'][0]
# Extract the current staging model MAPE
current_model_mape = mlflow.get_run(current_model['run_id']).data.metrics[model_config['COMPARISON_METRIC']]
# Get the new model MAPE
candidate_model_mape = mlflow.get_run(result.run_id).data.metrics[model_config['COMPARISON_METRIC']]

# COMMAND ----------

# MAGIC %md
# MAGIC If the newest run beats the oldest, then transition the last production version to Archived

# COMMAND ----------

if candidate_model_mape < current_model_mape:
    print(f"Candidate model has a better {model_config['COMPARISON_METRIC']} than the active model. Switching models...")
    
    client.transition_model_version_stage(
        name=model_config['REGISTER_MODEL_NAME'],
        version=result.version,
        stage='Staging',
    )

    client.transition_model_version_stage(
        name=model_config['REGISTER_MODEL_NAME'],
        version=current_model['version'],
        stage='Archived',
    )
else:
    print(f"Active model has a better {model_config['COMPARISON_METRIC']} than the candidate model. No changes to be applied.")
    
print(f"Candidate: {model_config['COMPARISON_METRIC']} = {candidate_model_mape}\nCurrent: = {current_model_mape}")

# COMMAND ----------

# Get the model that is now in stage
model_staging_uri = "models:/{model_name}/staging".format(model_name=model_config['REGISTER_MODEL_NAME'])

# Now load the new model to make predictions
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_staging_uri))
model_staging = mlflow.sklearn.load_model(model_staging_uri)

# COMMAND ----------

model_staging

# COMMAND ----------


