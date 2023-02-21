# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

import sys

sys.path.insert(0,'.')

from src.config import *



def train_model(X_train: pd.DataFrame,  y_train: pd.DataFrame, params: dict = None):
    """
    Trains a XGBoost model for Forecasting
    
    :param X_train: Training Features
    :param y_train: Training Target

    :return: Fitted model
    """
    logger.info("Training the model...")

    # create the model
    if params is None:
        print("param none")
        xgboost_model = xgb.XGBRegressor()

    else:
        # use existing params
        xgboost_model = xgb.XGBRegressor(
            **params
            )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        eval_metric=["rmse", "logloss"],
        )

    # save model
    dump(xgboost_model, f"./models/{model_config['REGISTER_MODEL_NAME_EVAL']}.joblib")



    return xgboost_model


def validate_model_stepwise(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, stock_name: str) -> pd.DataFrame:
    """
    Make predictions for the next `forecast_horizon` days using a XGBoost model
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas dataframe): The target data
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        None
    """

    logger.info("Starting the pipeline..")


    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []


    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # So we need to start at training = training until the total forecast horizon, then, perform the next step
    # After forecasting the next step, we need to append the new line to the training dataset and so on

    for day in range(forecast_horizon, 0, -1):

        # update the training and testing sets
        X_train = X.iloc[:-day, :]
        y_train = y.iloc[:-day]
 
        if day != 1:
            # the testing set will be the next day after the training
            X_test = X.iloc[-day:-day+1,:]
            y_test = y.iloc[-day:-day+1]

        else:
            # need to change the syntax for the last day (for -1:-2 will not work)
            X_test = X.iloc[-day:,:]
            y_test = y.iloc[-day:]


        # only the first iteration will use the true value of y_train
        # because the following ones will use the last predicted value as true value
        # so we simulate the process of predicting out-of-sample
        if len(predictions) != 0:
            # update the y_train with the last predictions
            y_train.iloc[-len(predictions):] = predictions[-len(predictions):]

            # now update the Close_lag_1 feature
            X_train.iloc[-len(predictions):, -1] = y_train.shift(1).iloc[-len(predictions):]
            X_train = X_train.dropna()

        else:
            pass
        
        xgboost_model = load(f"./models/{stock_name}_xgb.joblib")
        parameters = xgboost_model.get_xgb_params()
        # train the model
        xgboost_model = train_model(X_train.drop("Date", axis=1), y_train, params=parameters)

        # make prediction
        prediction = xgboost_model.predict(X_test.drop("Date", axis=1))

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["Date"].max())

    # Calculate the resulting metric
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
 
    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["Date", 'Actual', 'Forecast'])
    pred_df["Forecast"] = pred_df["Forecast"].astype("float64")
    #visualize_validation_results(pred_df, model_mape, model_rmse)

    return pred_df


def make_predict(model, forecast_horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:

    """
    Make predictions for the next `forecast_horizon` days using a XGBoost model
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas dataframe): The target data
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        None
    """

    future_df_feat = future_df.copy()

    # Create empty list for storing each prediction
    predictions = []

    for day in range(0, forecast_horizon):

        # extract the next day to predict
        x_inference = pd.DataFrame(future_df_feat.drop("Date", axis=1).loc[day, :]).transpose()
        prediction = model.predict(x_inference)[0]
        predictions.append(prediction)

        # get the prediction and input as the lag 1
        if day != forecast_horizon-1:
        
            future_df_feat.loc[day+1, "Close_lag_1"] = prediction

        else:
            # check if it is the last day, so we stop
            break

    future_df_feat["Forecast"] = predictions
    future_df_feat["Forecast"] = future_df_feat["Forecast"].astype('float64')
    future_df_feat = future_df_feat[["Date", "Forecast"]].copy()
    return future_df_feat


def time_series_grid_search_xgb(X, y, param_grid, stock_name, n_splits=5, random_state=0):
    """
    Performs time series hyperparameter tuning on an XGBoost model using grid search.
    
    Parameters:
    - X (pd.DataFrame): The input feature data
    - y (pd.Series): The target values
    - param_grid (dict): Dictionary of hyperparameters to search over
    - n_splits (int): Number of folds for cross-validation (default: 5)
    - random_state (int): Seed for the random number generator (default: 0)
    
    Returns:
    - best_model (xgb.XGBRegressor): The best XGBoost model found by the grid search
    - best_params (dict): The best hyperparameters found by the grid search
    """

    # perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = xgb.XGBRegressor(random_state=random_state)
    grid_search = GridSearchCV(model, param_grid, cv=tscv, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # save the best model
    dump(best_model, f"./models/{stock_name}_xgb.joblib")
    return best_model, best_params


def predictions_sanity_check(client, run_info, y_train, pred_df, model_mape, stage_version):
    """
    Check if the predictions are reliable.
    """

    newest_run_id = run_info.run_id
    newest_run_name = run_info.run_name

    # register the model
    logger.debug("Registering the model...")
    model_details = mlflow.register_model(
        model_uri = f"runs:/{newest_run_id}/{newest_run_name}",
        name = model_config[f'REGISTER_MODEL_NAME_{stage_version}']
    )

    # validate the predictions
    # check if the MAPE is less than 3%
    # check if the predictions have similar variation of historical
    logger.debug("Checking if the metrics and forecasts are valid...")
    if (model_mape < 0.03) and (0 < pred_df["Forecast"].std() < y_train.std()*1.5):

        # if so, transit to staging
        client.transition_model_version_stage(
            name=model_config[f'REGISTER_MODEL_NAME_{stage_version}'],
            version=model_details.version,
            stage='Staging',
        )

        # return the model details
        logger.debug("The model is valid. Returning the model details...")
        return model_details

    else:
        # if not, discard it
        client.delete_model_version(
            name=model_config[f'REGISTER_MODEL_NAME_{stage_version}'],
            version=model_details.version,
        )
        
        # return false to indicate that the model is not valid
        return False


def compare_models(client, model_details, stage_version) -> None:

    # get the metrics of the Production model
    models_versions = []

    for mv in client.search_model_versions("name='{}'".format(model_config[f'REGISTER_MODEL_NAME_{stage_version}'])):
        models_versions.append(dict(mv))

    current_prod_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]

    # Extract the current staging model MAPE
    current_model_mape = mlflow.get_run(current_prod_model['run_id']).data.metrics[model_config['VALIDATION_METRIC']]

    # Get the new model MAPE
    candidate_model_mape = mlflow.get_run(model_details.run_id).data.metrics[model_config['VALIDATION_METRIC']]

    # compare the MAPEs
    print('\n')
    print("-"*10 + " Continous Deployment Results " + "-"*10)

    if candidate_model_mape < current_model_mape:

        print(f"Candidate model has a better or equal {model_config['VALIDATION_METRIC']} than the active model. Switching models...")
        
        # archive the previous version
        client.transition_model_version_stage(
            name=model_config[f'REGISTER_MODEL_NAME_{stage_version}'],
            version=current_prod_model['version'],
            stage='Archived',
        )

        # transition the newest version
        client.transition_model_version_stage(
            name=model_config[f'REGISTER_MODEL_NAME_{stage_version}'],
            version=model_details.version,
            stage='Production',
        )


    else:
        print(f"Active model has a better {model_config['VALIDATION_METRIC']} than the candidate model.\nTransiting the new staging model to None.")
        
        client.transition_model_version_stage(
            name=model_config[f'REGISTER_MODEL_NAME_{stage_version}'],
            version=model_details.version,
            stage='None',
        )
        
    print(f"Candidate {model_config['VALIDATION_METRIC']}: {candidate_model_mape}\nCurrent {model_config['VALIDATION_METRIC']}: {current_model_mape}")
    print("-"*50)
    print('\n')


def cd_pipeline(run_info, y_train, pred_df, model_mape):

    logger.debug(" ----- Starting CD pipeline -----")
    
    # create a new Mlflow client
    client = MlflowClient()

    # validate the predictions
    model_details = predictions_sanity_check(client, run_info, y_train, pred_df, model_mape, "VAL")

    if model_details:
        # compare the new model with the production model
        logger.info("The model is reliable. Comparing it with the production model...")
        compare_models(client, model_details, "VAL")

    else:
        logger.info("The model is not reliable. Discarding it.")

    logger.debug(" ----- CD pipeline finished -----")

