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

RUN_NAME = 'XGBoost_Hyperopt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Build the hyperparameter optimisation

# COMMAND ----------

def objective(search_space):
    
    model = XGBRegressor(
        subsample= xgboost_fixed_model_config['SUBSAMPLE'],
        seed = xgboost_fixed_model_config['SEED'],
        **search_space
    )
    
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds = 200,
        eval_metric = ['rmse'],
        eval_set=[[X_train, y_train],[X_test, y_test]]
    )
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    
    return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

search_space = xgboost_hyperparameter_config

algorithm = tpe.suggest

spark_trials = SparkTrials(parallelism=PARALELISM)

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

xgboost_best_param_names = space_eval(search_space, best_params)

# COMMAND ----------

xgboost_best_param_names

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

X_train

# COMMAND ----------

with mlflow.start_run(run_name = RUN_NAME) as run:
    # First define the fixed parameters
    seed = xgboost_fixed_model_config['SEED']
    subsample = xgboost_fixed_model_config['SUBSAMPLE']
    
    # Getting the best parameters configuration
    try:
        max_depth = xgboost_best_param_names['max_depth']
        learning_rate = xgboost_best_param_names['learning_rate']
        gamma = xgboost_best_param_names['gamma']
        reg_lambda = xgboost_best_param_names['reg_lambda']
        n_estimators = xgboost_best_param_names['n_estimators']
        scale_pos_weight = xgboost_best_param_names['scale_pos_weight']
        colsample_bytree = xgboost_best_param_names['colsample_bytree']
        
    # If something goes wrong, select the pre-selected parameters in the config file
    except:
        max_depth = xgboost_hyperparameter_config['MAX_DEPTH']
        learning_rate = xgboost_hyperparameter_config['LEARNING_RATE']
        gamma = xgboost_hyperparameter_config['gamma']
        reg_lambda = xgboost_hyperparameter_config['reg_lambda']
        n_estimators = xgboost_hyperparameter_config['n_estimators']
        scale_pos_weight = xgboost_hyperparameter_config['scale_pos_weight']
        colsample_bytree = xgboost_hyperparameter_config['colsample_bytree']

    # Create the model instance if the selected parameters
    model = XGBRegressor(
        max_depth = max_depth,
        learning_rate = learning_rate,
        gamma = gamma,
        reg_lambda = reg_lambda,
        n_estimators = n_estimators,
        scale_pos_weight = scale_pos_weight,
        colsample_bytree = colsample_bytree,
        seed = seed,
        subsample = subsample
    )

    # Training the model
    model_fit = model.fit(
        X=X_train,
        y=y_train,
        early_stopping_rounds=100,
        eval_metric= ['mape'],
        eval_set =[(X_train, y_train), (X_test, y_test)]
    )

    ### Perform Predictions
    # Use the model to make predictions on the test dataset.
    predictions = model_fit.predict(X_val)

    ### Log the metrics

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("reg_lambda", reg_lambda)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("colsample_bytree", colsample_bytree)

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
    axs.set_title(f"XGBoost Predicted versus ground truth\n R2 = {r2} | RMSE = {rmse} | MAPE = {mape}")
    axs.set_xlabel(f"True {TARGET_VARIABLE}")
    axs.set_ylabel(f"Predicted {TARGET_VARIABLE}")
    plt.savefig("artefacts/scatter_plot_xgboost.png")
    fig.show()

    mlflow.log_artifact("artefacts/scatter_plot_xgboost.png")

    mlflow.sklearn.log_model(model_fit, RUN_NAME)

    np.savetxt('artefacts/predictions_xgboost.csv', predictions, delimiter=',')

    # Log the saved table as an artifact
    mlflow.log_artifact("artefacts/predictions_xgboost.csv")

    # Convert the residuals to a pandas dataframe to take advantage of graphics  
    predictions_df = pd.DataFrame(data = predictions - y_val)

    plt.figure()
    plt.plot(predictions_df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")

    plt.savefig("artefacts/residuals_plot_xgboost.png")
    mlflow.log_artifact("artefacts/residuals_plot_xgboost.png")

# COMMAND ----------

model.predict(X_test)

# COMMAND ----------

model.feature_names_in_

# COMMAND ----------


