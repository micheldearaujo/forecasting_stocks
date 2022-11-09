# Databricks notebook source
# MAGIC %md
# MAGIC ## Modeling a simple Random Forest using MLflow

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
X = train_df.drop(TARGET_VARIABLE, axis=1)
y = train_df[TARGET_VARIABLE]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

print(mape, r2)

# COMMAND ----------

fig, axs = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
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
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(mape, r2)
    

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

with mlflow.start_run(run_name = RUN_NAME) as run:
    # First configure the fixed parameters, such as random_state
    random_state = random_forest_fixed_model_config['RANDOM_STATE']
    
    # Getting the best parameters configuration
    try:
        bootstrap = rf_best_param_names['bootstrap']
        max_depth = rf_best_param_names['max_depth']
        max_features = rf_best_param_names['max_features']
        min_samples_leaf = rf_best_param_names['min_samples_leaf']
        min_samples_split = rf_best_param_names['min_samples_split']
        n_estimators = rf_best_param_names['n_estimators']
        
    # If something goes wrong, select the pre-selected parameters in the config file
    except:
        bootstrap = random_forest_model_config['bootstrap']
        max_depth = random_forest_model_config['max_depth']
        max_features = random_forest_model_config['max_features']
        min_samples_leaf = random_forest_model_config['min_samples_leaf']
        min_samples_split = random_forest_model_config['min_samples_split']
        n_estimators = random_forest_model_config['n_estimators']
 
        

    # Create the model instance if the selected parameters
    model = RandomForestRegressor(
        bootstrap = bootstrap,
        max_depth = max_depth,
        max_features = max_features,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split,
        n_estimators = n_estimators,
        random_state = random_state,
    )

    # Training the model
    model_fit = model.fit(
        X=X_train,
        y=y_train
    )

    ### Perform Predictions
    # Use the model to make predictions on the test dataset.
    predictions = model_fit.predict(X_val)

    ### Log the metrics

    mlflow.log_param("bootstrap", bootstrap)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("n_estimators", n_estimators)

    # Define a metric to use to evaluate the model.

    # RMSE
    rmse = round(np.sqrt(mean_squared_error(y_val, predictions)), 2)
    # R2
    r2 = round(r2_score(y_val, predictions), 2)
    # R2 adjusted
    p = X_val.shape[1]
    n = X_val.shape[0]
    adjust_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    # MAPE
    mape = round(mean_absolute_percentage_error(y_val, predictions), 3)


    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("R2_Adj", adjust_r2)
    mlflow.log_metric("MAPE", mape)

    mlflow.log_metric('Dataset_Size', df.shape[0])
    mlflow.log_metric('Number_of_variables', X_train.shape[1])

    fig, axs = plt.subplots(figsize=(12, 8))
    axs.scatter(x=y_val, y=predictions)
    axs.set_title(f"Random Forest Predicted versus ground truth\n R2 = {r2} | RMSE = {rmse} | MAPE = {mape}")
    axs.set_xlabel(f"True {TARGET_VARIABLE}")
    axs.set_ylabel(f"Predicted {TARGET_VARIABLE}")
    plt.savefig("artefacts/scatter_plot_rf.png")
    fig.show()

    mlflow.log_artifact("artefacts/scatter_plot_rf.png")

    mlflow.sklearn.log_model(model_fit, RUN_NAME)

    np.savetxt('artefacts/predictions_rf.csv', predictions, delimiter=',')

    # Log the saved table as an artifact
    mlflow.log_artifact("artefacts/predictions_rf.csv")

    # Convert the residuals to a pandas dataframe to take advantage of graphics  
    predictions_df = pd.DataFrame(data = predictions - y_val)

    plt.figure()
    plt.plot(predictions_df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")

    plt.savefig("artefacts/residuals_plot_rf.png")
    mlflow.log_artifact("artefacts/residuals_plot_rf.png")

# COMMAND ----------

model

# COMMAND ----------


