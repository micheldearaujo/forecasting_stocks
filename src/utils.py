# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

from config import *


def build_features(raw_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    This function creates the features for the dataset to be consumed by the
    model
    
    :param raw_df: Raw Pandas DataFrame to create the features of
    :param features_list: The list of features to create

    :return: Pandas DataFrame with the new features
    """

    logger.info("Building the features...")

    stock_df_featurized = raw_df.copy()
    for feature in features_list:
        
        # Create "Time" features]
        if feature == "day_of_month":
            stock_df_featurized['day_of_month'] = stock_df_featurized["Date"].apply(lambda x: x.day)
        elif feature == "month":
            stock_df_featurized['month'] = stock_df_featurized['Date'].apply(lambda x: x.month)
        elif feature == "quarter":
            stock_df_featurized['quarter'] = stock_df_featurized['Date'].apply(lambda x: x.quarter)

    # Create "Lag" features
    # The lag 1 feature will become the main regressor, and the regular "Close" will become the target.
    # As we saw that the lag 1 holds the most aucorrelation, it is reasonable to use it as the main regressor.
        elif feature == "Close_lag_1":
            stock_df_featurized['Close_lag_1'] = stock_df_featurized['Close'].shift()


    # Drop nan values because of the shift
    stock_df_featurized = stock_df_featurized.dropna()
    # Drop the Date column
    #stock_df_featurized = stock_df_featurized.drop("Date", axis=1)
    # Save the dataset
    stock_df_featurized.to_csv("./data/processed/processed_stock_prices.csv", index=False)

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


def visualize_validation_results(pred_df: pd.DataFrame, model_mape: float, model_rmse: float):
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

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']} days Forecast for {STOCK_NAME}\nMAPE: {round(model_mape*100, 2)}% | RMSE: R${model_rmse}")
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    plt.savefig(f"./reports/figures/XGBoost_predictions_{dt.datetime.now()}.png")
    plt.show()


def train_model(X_train: pd.DataFrame,  y_train: pd.DataFrame, random_state:int=42):
    """
    Trains a XGBoost model for Forecasting
    
    :param X_train: Training Features
    :param y_train: Training Target

    :return: Fitted model
    """
    logger.info("Training the model...")

    # create the model
    xgboost_model = xgb.XGBRegressor(
        random_state=random_state,
        )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
        )

    return xgboost_model


def make_out_of_sample_predictions(X:pd.DataFrame, y:pd.Series, forecast_horizon: int) -> pd.DataFrame:
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
        
        
        # train the model
        xgboost_model = train_model(X_train.drop("Date", axis=1), y_train)

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
    print(pred_df)
    #visualize_validation_results(pred_df, model_mape, model_rmse)

    return pred_df


