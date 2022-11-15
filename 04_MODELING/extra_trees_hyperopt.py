# Databricks notebook source
# MAGIC %md
# MAGIC ## Extra Trees Hyperparameter Optimisation
# MAGIC 
# MAGIC **Objective**: This notebook's objective is train and optimise a Extra Trees regression model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

RUN_NAME = 'ExtraTrees_Hyperopt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()
X = train_df.drop(model_config['TARGET_VARIABLE'], axis=1)
y = train_df[model_config['TARGET_VARIABLE']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Build the hyperparameter optimisation

# COMMAND ----------

def objective(search_space):
    
    model = ExtraTreesRegressor(
        random_state=42,
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

search_space = etr_hyperparameter_config

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

etr_best_param_names = space_eval(search_space, best_params)

# COMMAND ----------

etr_best_param_names

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

with mlflow.start_run(run_name = RUN_NAME) as run:
    
    seed = xgboost_model_config['SEED']
    subsample = xgboost_model_config['SUBSAMPLE']
    
    # Getting the best parameters configuration
    try:
        criterion = etr_best_param_names['criterion']
        n_estimators = etr_best_param_names['n_estimators']
        max_depth = etr_best_param_names['max_depth']
        min_samples_leaf = etr_best_param_names['min_samples_leaf']
        min_samples_split = etr_best_param_names['min_samples_split']
        
    # If something goes wrong, select the pre-selected parameters in the config file
    except:
        criterion = etr_hyperparameter_config['criterion']
        n_estimators = etr_hyperparameter_config['n_estimators']
        max_depth = etr_hyperparameter_config['max_depth']
        min_samples_leaf = etr_hyperparameter_config['min_samples_leaf']
        min_samples_split = etr_hyperparameter_config['min_samples_split']

    # Create the model instance if the selected parameters
    model = ExtraTreesRegressor(
        criterion = criterion,
        max_depth = max_depth,
        n_estimators = n_estimators,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split,
    )

    # Training the model
    model_fit = model.fit(
        X=X_train,
        y=y_train,
    )
    
    # Log the model
    mlflow.sklearn.log_model(model_fit, RUN_NAME)

    ### Perform Predictions
    # Use the model to make predictions on the test dataset.
    predictions = model_fit.predict(X_test)

    ### Log the metrics
    mlflow.log_param("criterion", criterion)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("min_samples_split", min_samples_split)

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
    
    # Log the metrics
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("R2_Adj", adjust_r2)
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric('Dataset_Size', X_train.shape[0])
    mlflow.log_metric('Number_of_variables', X_train.shape[1])
    
    # Plot the results
    fig, axs = plt.subplots(figsize=(12, 8))
    axs.scatter(x=y_test, y=predictions)
    axs.set_title(f"ETR Predicted versus ground truth\n R2 = {r2} | RMSE = {rmse} | MAPE = {mape}")
    axs.set_xlabel(f"True {TARGET_VARIABLE}")
    axs.set_ylabel(f"Predicted {TARGET_VARIABLE}")
    plt.savefig("artifacts/scatter_plot_etr.png")
    fig.show()

    np.savetxt('artifacts/predictions_etr.csv', predictions, delimiter=',')
    # Convert the residuals to a pandas dataframe to take advantage of graphics  
    predictions_df = pd.DataFrame(data = predictions - y_test)

    plt.figure()
    plt.plot(predictions_df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")
    plt.savefig("artifacts/residuals_plot_etr.png")
    
    # Log some artifacts
    mlflow.log_artifact("artifacts/scatter_plot_etr.png")
    mlflow.log_artifact("artifacts/residuals_plot_etr.png")
    mlflow.log_artifact("artifacts/predictions_etr.csv")
