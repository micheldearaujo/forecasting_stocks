# Databricks notebook source
# MAGIC %md
# MAGIC ## XGBoost Hyperparameter Optimisation
# MAGIC 
# MAGIC **Objective**: This notebook's objective is train and optimise a XGBoost regression model
# MAGIC 
# MAGIC **Takeaways**: The key takeaways of this notebook are:

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

df = spark.sql("SELECT * FROM default.fish_cleaned_testing").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Split the dataset

# COMMAND ----------

X = df.drop(TARGET_VARIABLE, axis=1)
y = df[TARGET_VARIABLE]

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Make Predictions

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Load the best model (Staging model)

# COMMAND ----------

# Get the model that is now in stage
staging_model_uri = "models:/{model_name}/staging".format(model_name=REGISTER_MODEL_NAME)

# Now load the new model to make predictions
print("Loading registered model version from URI: '{model_uri}'".format(model_uri=staging_model_uri))
staging_model = mlflow.sklearn.load_model(staging_model_uri)

# COMMAND ----------

staging_model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Make predictions

# COMMAND ----------

predictions = staging_model.predict(X_test)

# COMMAND ----------


