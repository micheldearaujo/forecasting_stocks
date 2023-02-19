# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

import sys

sys.path.insert(0,'.')

from src.config import *


def make_dataset(stock_name: str, period: str, interval: str):
    """
    Creates a dataset of the closing prices of a given stock.
    
    Parameters:
        stock_name (str): The name of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing prices of the given stock.
    """

    stock_price_df = yfin.Ticker(stock_name).history(period=period, interval=interval)
    stock_price_df['Stock'] = stock_name
    stock_price_df = stock_price_df[['Close']]
    stock_price_df = stock_price_df.reset_index()
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df['Date'] = stock_price_df['Date'].apply(lambda x: x.date())
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df.to_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), index=False)#'./data/raw/raw_stock_prices.csv', index=False)

    return stock_price_df


def build_features(raw_df: pd.DataFrame, features_list: list, save: bool=True) -> pd.DataFrame:
    """
    This function creates the features for the dataset to be consumed by the
    model
    
    :param raw_df: Raw Pandas DataFrame to create the features of
    :param features_list: The list of features to create

    :return: Pandas DataFrame with the new features
    """

    logger.debug("Started building features...")
    stock_df_featurized = raw_df.copy()
    for feature in features_list:
        
        # create "Time" features
        if feature == "day_of_month":
            stock_df_featurized['day_of_month'] = stock_df_featurized["Date"].apply(lambda x: float(x.day))
        elif feature == "month":
            stock_df_featurized['month'] = stock_df_featurized['Date'].apply(lambda x: float(x.month))
        elif feature == "quarter":
            stock_df_featurized['quarter'] = stock_df_featurized['Date'].apply(lambda x: float(x.quarter))

    # Create "Lag" features
    # The lag 1 feature will become the main regressor, and the regular "Close" will become the target.
    # As we saw that the lag 1 holds the most aucorrelation, it is reasonable to use it as the main regressor.
        elif feature == "Close_lag_1":
            stock_df_featurized['Close_lag_1'] = stock_df_featurized['Close'].shift()


    # Drop nan values because of the shift
    stock_df_featurized = stock_df_featurized.dropna()
    try:
        logger.debug("Rounding the features to 2 decimal places...")
        # handle exception when building the future dataset
        stock_df_featurized['Close'] = stock_df_featurized['Close'].apply(lambda x: round(x, 2))
        stock_df_featurized['Close_lag_1'] = stock_df_featurized['Close_lag_1'].apply(lambda x: round(x, 2))
    except KeyError:
        pass
    

    # Save the dataset
    #stock_df_featurized.to_csv("./data/processed/processed_stock_prices.csv", index=False)
    if save:
        stock_df_featurized.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), index=False)

    logger.debug("Features built successfully!")

    return stock_df_featurized


def ts_train_test_split(data: pd.DataFrame, target:str, test_size: int):
    """
    Splits the Pandas DataFrame into training and tests sets
    based on a Forecast Horizon value.

    Paramteres:
        data (pandas dataframe): Complete dataframe with full data
        targer (string): the target column name
        test_size (int): the amount of periods to forecast

    Returns:
        X_train, X_test, y_train, y_test dataframes for training and testing
    """

    logger.info("Spliting the dataset...")

    train_df = data.iloc[:-test_size, :]
    test_df = data.iloc[-test_size:, :]
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


def visualize_validation_results(pred_df: pd.DataFrame, model_mape: float, model_rmse: float, stock_name: str):
    """
    Creates visualizations of the model validation

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        model_mape: The validation MAPE
        model_rmse: The validation RMSE

    Returns:
        None
    """

    logger.info("Vizualizing the results...")

    fig, axs = plt.subplots(figsize=(12, 5))
    # Plot the Actuals
    sns.lineplot(
        data=pred_df,
        x="Date",
        y="Actual",
        label="Testing values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Actual",
        ax=axs,
        size="Actual",
        sizes=(80, 80), legend=False
    )

    # Plot the Forecasts
    sns.lineplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        label="Forecast values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        ax=axs,
        size="Forecast",
        sizes=(80, 80), legend=False
    )

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']} days Forecast for {stock_name}\nMAPE: {round(model_mape*100, 2)}% | RMSE: R${model_rmse}")
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    plt.savefig(f"./reports/figures/XGBoost_predictions_{dt.datetime.now().date()}.png")
    #plt.show()
    return fig


def visualize_forecast(pred_df: pd.DataFrame, historical_df: pd.DataFrame, stock_name: str):
    """
    Creates visualizations of the model forecast

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        historical_df: DataFrame with historical values

    Returns:
        None
    """

    logger.info("Vizualizing the results...")

    fig, axs = plt.subplots(figsize=(12, 5), dpi = 2000)
    # Plot the Actuals
    sns.lineplot(
        data=historical_df,
        x="Date",
        y="Close",
        label="Historical values",
        ax=axs
    )
    sns.scatterplot(
        data=historical_df,
        x="Date",
        y="Close",
        ax=axs,
        size="Close",
        sizes=(80, 80),
        legend=False
    )

    # Plot the Forecasts
    sns.lineplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        label="Forecast values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        ax=axs,
        size="Forecast",
        sizes=(80, 80),
        legend=False
    )

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']-4} days Forecast for {stock_name}")
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    #plt.show()
    return fig


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


def stepwise_forecasting(model, X, y, forecast_horizon):
    predictions = []
    actuals = []
    dates = []
    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # After forecasting the next step, we need to update the "lag" features with the last forecasted
    # value
    for day in range(forecast_horizon-4, 0, -1):
        
        if day != 1:
            # the testing set will be the next day after the training and we use the complete dataset
            X_test = X.iloc[-day:-day+1,:]
            y_test = y.iloc[-day:-day+1]

        else:
            # need to change the syntax for the last day (for -1:-2 will not work)
            X_test = X.iloc[-day:,:]
            y_test = y.iloc[-day:]

        # only the first iteration will use the true value of Close_lag_1
        # because the following ones will use the last predicted value as true value
        # so we simulate the process of predicting out-of-sample
        if len(predictions) != 0:
            
            # we need to update the X_test["Close_lag_1"] value, because
            # it should be equal to the last prediction (the "yesterday" value)
            X_test.iat[0, -1] = predictions[-1]            

        else:
            pass

        # make prediction
        prediction = model.predict(X_test)

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])

    # Calculate the resulting metric
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
    model_mae = round(mean_absolute_error(actuals, predictions), 2)

    return model_mape, model_rmse, model_mae


def validade_model_one_shot(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, stock_name: str) -> pd.DataFrame:
    """
    Make predictions for the next `forecast_horizon` days using a XGBoost model.
    This model is validated using One Shot Training, it means that we train the model
    once, and them perform the `forecast_horizon` predictions only loading the mdoel.
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas dataframe): The target data
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        pred_df: Pandas DataFrame with the forecasted values
    """

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    
    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon, :]
    y_train = y.iloc[:-forecast_horizon]
    
    # load the best model
    xgboost_model = load(f"./models/{stock_name}_params.joblib")
    # get the best parameters
    parameters = xgboost_model.get_xgb_params()

    # start the mlflow tracking
    with mlflow.start_run(run_name="model_validation") as run:

        # fit the model again with the best parameters
        print(parameters)

        xgboost_model = xgb.XGBRegressor(
            **parameters
        )
        

        # train the model
        xgboost_model.fit(
            X_train.drop("Date", axis=1),
            y_train,
            eval_set=[(X_train.drop("Date", axis=1), y_train)],
            eval_metric=["rmse", "logloss"],

        )

        # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
        # After forecasting the next step, we need to update the "lag" features with the last forecasted
        # value
        for day in range(forecast_horizon-4, 0, -1):
            
            if day != 1:
                # the testing set will be the next day after the training and we use the complete dataset
                X_test = X.iloc[-day:-day+1,:]
                y_test = y.iloc[-day:-day+1]

            else:
                # need to change the syntax for the last day (for -1:-2 will not work)
                X_test = X.iloc[-day:,:]
                y_test = y.iloc[-day:]

            # only the first iteration will use the true value of Close_lag_1
            # because the following ones will use the last predicted value as true value
            # so we simulate the process of predicting out-of-sample
            if len(predictions) != 0:
                
                # we need to update the X_test["Close_lag_1"] value, because
                # it should be equal to the last prediction (the "yesterday" value)
                X_test.iat[0, -1] = predictions[-1]            

            else:
                pass

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
        fig = visualize_validation_results(pred_df, model_mape, model_rmse, stock_name)
        # get the training results
        learning_results = xgboost_model.evals_result()
      
        # Plotting the Learning Results
        fig2, axs = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle("XGBoost Learning Curves")
        axs[0].plot(learning_results['validation_0']['rmse'], label='Training')
        axs[0].set_title("RMSE Metric")
        axs[0].set_ylabel("RMSE")
        axs[0].set_xlabel("Iterations")
        axs[0].legend()

        axs[1].plot(learning_results['validation_0']['logloss'], label='Training')
        axs[1].set_title("Logloss Metric")
        axs[1].set_ylabel("Logloss")
        axs[1].set_xlabel("Iterations")
        axs[1].legend()
        #plt.show()
        # ---- logging ----

        # log the parameters
        mlflow.log_params(parameters)

        # log the metrics
        mlflow.log_metric("MAPE", model_mape)
        mlflow.log_metric("RMSE", model_rmse)

        # log the figure
        mlflow.log_figure(fig, "validation_results.png")
        mlflow.log_figure(fig2, "learning_curves.png")

        # get model signature
        model_signature = infer_signature(X_train, pd.DataFrame(y_train))

        # log the model to mlflow
        mlflow.xgboost.log_model(
            xgb_model=xgboost_model,
            artifact_path="xgboost_model",
            input_example=X_train.head(),
            signature=model_signature
        )

        # execute the CD pipeline
        cd_pipeline(run.info, y_train, pred_df, model_mape)

    
    return pred_df


def make_future_df(forecast_horzion: int, model_df: pd.DataFrame, features_list: list):
    """
    Create a future dataframe for forecasting.

    Parameters:
        forecast_horizon (int): The number of days to forecast into the future.
        model_df (pandas dataframe): The dataframe containing the training data.

    Returns:
        future_df (pandas dataframe): The future dataframe used for forecasting.
    """

    # create the future dataframe with the specified number of days
    last_training_day = model_df.Date.max()
    date_list = [last_training_day + dt.timedelta(days=x+1) for x in range(forecast_horzion)]
    future_df = pd.DataFrame({"Date": date_list})

    # build the features for the future dataframe using the specified features
    inference_features_list = features_list[:-1]
    future_df = build_features(future_df, inference_features_list, save=False)

    # filter out weekends from the future dataframe
    future_df["day_of_week"] = future_df.Date.apply(lambda x: x.day_name())
    future_df = future_df[future_df["day_of_week"].isin(["Sunday", "Saturday"]) == False]
    future_df = future_df.drop("day_of_week", axis=1)
    future_df = future_df.reset_index(drop=True)

    # Ensure the data types of the features are correct
    #for feature in inference_features_list:
    #    future_df[feature] = future_df[feature].astype("int")
    
    # set the first lagged price value to the last price from the training data
    future_df["Close_lag_1"] = 0
    future_df.loc[future_df.index.min(), "Close_lag_1"] = model_df[model_df["Date"] == last_training_day]['Close'].values[0]
    return future_df


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

    # load the model and predict
    #model = load(os.path.join(MODELS_PATH, f"{model_config['REGISTER_MODEL_NAME']}.joblib"))
    
    # load the model using mlflow

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


def compare_models(client, model_details, stage_version):

    # get the metrics of the Production model
    models_versions = []

    for mv in client.search_model_versions("name='{}'".format(model_config[f'REGISTER_MODEL_NAME_{stage_version}'])):
        models_versions.append(dict(mv))

    current_prod_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]

    # Extract the current staging model MAPE
    current_model_mape = mlflow.get_run(current_prod_model['run_id']).data.metrics[model_config['VALIDATION_METRIC']]

    # Get the new model MAPE
    candidate_model_mape = mlflow.get_run(model_details.run_id).data.metrics[model_config['VALIDATION_METRIC']]

    # compare
    if candidate_model_mape < current_model_mape:
        print(f"Candidate model has a better {model_config['VALIDATION_METRIC']} than the active model. Switching models...")
        
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
        print(f"Active model has a better {model_config['VALIDATION_METRIC']} than the candidate model.\nTransiting the staging model to None.")
        
        client.transition_model_version_stage(
            name=model_config[f'REGISTER_MODEL_NAME_{stage_version}'],
            version=model_details.version,
            stage='None',
        )
        
    print(f"Candidate: {model_config['VALIDATION_METRIC']} = {candidate_model_mape}\nCurrent: = {current_model_mape}")


def cd_pipeline(run_info, y_train, pred_df, model_mape):

    logger.debug(" ----- Starting CD pipeline -----")
    
    client = MlflowClient()

    # validate the predictions
    model_details = predictions_sanity_check(client, run_info, y_train, pred_df, model_mape, "VAL")

    if model_details:
        # compare the new model with the production model
        logger.info("The model is reliable. Comparing it with the production model...")
        compare_models(client, model_details, "VAL")

    else:
        logger.info("The model is not reliable. Discarding it.")

    