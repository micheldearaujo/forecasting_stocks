# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

import sys

sys.path.insert(0,'.')

from src.config import *
from src.features.build_features import build_features



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

    plt.savefig(f"./reports/figures/XGBoost_predictions_{dt.datetime.now().date()}_{stock_name}.png")
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


def make_future_df(forecast_horzion: int, model_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
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
    
    # add stock column to iterate
    future_df["Stock"] = model_df.Stock.unique()[0]


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