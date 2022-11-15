# Databricks notebook source
# MAGIC %md
# MAGIC ## Random Forest Hyperparameter Optimisation
# MAGIC 
# MAGIC **Objective**: This notebook's objective is train and optimise a Random Forest regression model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

RUN_NAME = 'RandomForest_Hyperopt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Build the hyperparameter optimisation

# COMMAND ----------

def objective(search_space):
    
    model = RandomForestRegressor(
        random_state = random_forest_fixed_model_config['RANDOM_STATE'],
        **search_space
    )
    model.fit(
        X_train,
        y_train
    )
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

search_space = randomforest_hyperparameter_config

algorithm = tpe.suggest

spark_trials = SparkTrials(parallelism=model_config['PARALELISM'])

# COMMAND ----------

with mlflow.start_run(run_name=RUN_NAME):
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=algorithm,
        max_evals=model_config['MAX_EVALS'],
        trials=spark_trials
    )

# COMMAND ----------

rf_best_param_names = space_eval(search_space, best_params)

# COMMAND ----------

rf_best_param_names

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

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
    predictions = model_fit.predict(X_test)

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
    rmse = round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
    # R2
    r2 = round(r2_score(y_test, predictions), 2)
    # R2 adjusted
    p = X_test.shape[1]
    n = X_test.shape[0]
    adjust_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    # MAPE
    mape = round(mean_absolute_percentage_error(y_test, predictions), 3)


    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("R2_Adj", adjust_r2)
    mlflow.log_metric("MAPE", mape)

    mlflow.log_metric('Dataset_Size', X_train.shape[0])
    mlflow.log_metric('Number_of_variables', X_train.shape[1])

    fig, axs = plt.subplots(figsize=(12, 8))
    axs.scatter(x=y_test, y=predictions)
    axs.set_title(f"Random Forest Predicted versus ground truth\n R2 = {r2} | RMSE = {rmse} | MAPE = {mape}")
    axs.set_xlabel(f"True {TARGET_VARIABLE}")
    axs.set_ylabel(f"Predicted {TARGET_VARIABLE}")
    plt.savefig("artifacts/scatter_plot_rf.png")
    fig.show()

    mlflow.log_artifact("artifacts/scatter_plot_rf.png")

    mlflow.sklearn.log_model(model_fit, RUN_NAME)

    np.savetxt('artifacts/predictions_rf.csv', predictions, delimiter=',')

    # Log the saved table as an artifact
    mlflow.log_artifact("artifacts/predictions_rf.csv")

    # Convert the residuals to a pandas dataframe to take advantage of graphics  
    predictions_df = pd.DataFrame(data = predictions - y_test)

    plt.figure()
    plt.plot(predictions_df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")

    plt.savefig("artifacts/residuals_plot_rf.png")
    mlflow.log_artifact("artifacts/residuals_plot_rf.png")

# COMMAND ----------

model

# COMMAND ----------


