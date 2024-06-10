import sys
sys.path.insert(0,'.')

import re
import warnings
import yaml
import argparse
import logging
from typing import Any

import xgboost as xgb
import matplotlib.pyplot as plt

from src.utils import *
from src.features.feat_eng import create_date_features

warnings.filterwarnings("ignore")

with open("src/configuration/logging_config.yaml", 'r') as f:  

    loggin_config = yaml.safe_load(f.read())
    logging.config.dictConfig(loggin_config)

with open("src/configuration/project_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())

logger = logging.getLogger(__name__)

def load_production_model_sklearn(model_type, ticker_symbol):
    """
    Loading the Sklearn models saved using the traditional Joblib format.
    """
    MODELS_PATH = config['paths']['models_path']
    model_file_path = f"{MODELS_PATH}/{model_type}/Model_{ticker_symbol}.model"

    if model_type == 'xgb':
        current_prod_model = xgb.XGBRegressor()
        current_prod_model._Booster = xgb.Booster()
        current_prod_model._Booster.load_model(model_file_path)
    else:  
        current_prod_model = load(model_file_path)

    return current_prod_model


def initialize_lag_values(df: pd.DataFrame, features_list: list, target_column: str, future_df: pd.DataFrame):
    """Calculates and sets the initial lag feature value for a given lag and target column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the historical data.
        features_list (list): The modeling features list.
        target_column (str): The name of the target column.
        future_df (pd.DataFrame): DataFrame to store the future (out-of-sample) features.

    Returns:
        pd.DataFrame: The updated future_df with the calculated lag feature.
    """
    for feature in filter(lambda f: "LAG" in f, features_list):

        lag_value = int(feature.split("_")[-1])
        future_df.loc[future_df.index.min(), f"{target_column}_LAG_{lag_value}"] = df[target_column].iloc[-lag_value]
    return future_df


def initialize_ma_values(df: pd.DataFrame, features_list: list, target_column: str, future_df: pd.DataFrame):
    """Calculates and sets the initial moving average feature value for a given window size and target column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the historical data.
        features_list (list): The modeling features list.
        target_column (str): The name of the target column.
        future_df (pd.DataFrame): DataFrame to store the future (out-of-sample) features.

    Returns:
        pd.DataFrame: The updated future_df with the calculated moving average feature.
    """
    for feature in filter(lambda f: "MA" in f, features_list):

        ma_value = int(feature.split("_")[-1])
        future_df.loc[future_df.index.min(), f"{target_column}_MA_{ma_value}"] = (
            df[target_column].rolling(ma_value).mean().iloc[-1]
        )

    return future_df


def make_future_df(forecast_horzion: int, model_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    Create a future dataframe for forecasting.

    Parameters:
        forecast_horizon (int): The number of days to forecast into the future.
        model_df (pandas dataframe): The dataframe containing the training data.

    Returns:
        future_df (pandas dataframe): The future dataframe used for forecasting.
    """
    TARGET_NAME = config['model_config']['TARGET_NAME']

    # create the future dataframe with the specified number of days
    last_training_day = model_df["DATE"].max()
    date_list = [last_training_day + dt.timedelta(days=x+1) for x in range(forecast_horzion+1)]
    future_df = pd.DataFrame({"DATE": date_list})
    
    # add stock column to iterate
    future_df["STOCK"] = model_df["STOCK"].unique()[0]

    future_df = create_date_features(df=future_df)

    # filter out weekends from the future dataframe
    future_df = future_df[~future_df["DAY_OF_WEEK"].isin([5, 6])]

    future_df.reset_index(drop=True, inplace=True)

    future_df = initialize_ma_values(model_df, features_list, TARGET_NAME, future_df)
    future_df = initialize_lag_values(model_df, features_list, TARGET_NAME, future_df)
    
    
    # # set the first lagged price value to the last price from the training data
    # ma_and_lag_features = [feature for feature in features_list if "MA" in feature or "LAG" in feature]
    # for feature in ma_and_lag_features:
        
    #     future_df[feature] = 0
    #     if "MA" in feature:
    #         ma_value = int(feature.split("_")[-1])
    #         future_df.loc[future_df.index.min(), feature] = model_df[TARGET_NAME].rolling(ma_value).mean().values[-1]

    #     else:
    #         lag_value = int(feature.split("_")[-1])
    #         future_df.loc[future_df.index.min(), feature] = model_df[TARGET_NAME].values[-lag_value]
    
    future_df = future_df.reindex(columns=["DATE", "STOCK", *features_list])
    print(future_df.tail())
    return future_df


def make_predict(model: Any, forecast_horizon: int, future_df: pd.DataFrame, past_target_values: list) -> pd.DataFrame:

    """
    Make predictions for the next `forecast_horizon` days using a XGBoost model
    
    Parameters:
        model (sklearn model): Scikit-learn best model to use to perform inferece.
        forecast_horizon (int): The amount of days to predict into the future.
        future_df (pd.DataFrame): The "Feature" DataFrame (X) with future index.
        past_target_values (list): The target variable's historical values to calculate the moving averages on.
        
    Returns:
        pd.DataFrame: The future DataFrame with forecasts.
    """

    future_df_feat = future_df.copy()
    all_features = future_df_feat.columns
    predictions = []

    FH_WITHOUT_WEEKENDS = len(future_df_feat)

    for day in range(0, FH_WITHOUT_WEEKENDS):

        # extract the next day to predict
        X_inference = pd.DataFrame(future_df_feat.drop(columns=["DATE", "STOCK"]).loc[day, :]).transpose()
        print(X_inference)

        prediction = model.predict(X_inference)[0]
        predictions.append(prediction)
        
        # Append the prediction to the last_closing_prices
        past_target_values.append(prediction)

        # get the prediction and input as the lag 1
        if day != FH_WITHOUT_WEEKENDS-1:
            
            lag_features = [feature for feature in all_features if "LAG" in feature]
            for feature in lag_features:
                lag_value = int(feature.split("_")[-1])
                future_df_feat.loc[day+1, feature] = past_target_values[-lag_value]

            
            moving_averages_features = [feature for feature in all_features if "MA" in feature]
            for feature in moving_averages_features:
                ma_value = int(feature.split("_")[-1])
                last_n_closing_prices = [*past_target_values[-ma_value+1:], prediction]
                next_ma_value = np.mean(last_n_closing_prices)
                future_df_feat.loc[day+1, feature] = next_ma_value

        else:
            # check if it is the last day, so we stop
            break
    
    future_df_feat["Forecast"] = predictions
    future_df_feat["Forecast"] = future_df_feat["Forecast"].astype('float64')
    future_df_feat = future_df_feat[["DATE", "Forecast"]].copy()

    print(future_df_feat.tail())
    
    return future_df_feat



def inference_pipeline(model_type=None, ticker_symbol=None, write_to_table=True):
    """
    Run the inference pipeline for predicting stock prices using the production model.

    This function loads the featurized dataset, searches for the production model for a specific stock,
    and makes predictions for the future timeframe using the loaded model. The predictions are then
    saved to a CSV file.

    Parameters:
        None

    Returns:
        None
    """

    FORECAST_HORIZON = config['model_config']['forecast_horizon']
    available_models = config['model_config']['available_models']
    TARGET_NAME = config['model_config']['TARGET_NAME']
    features_list = config['features_list']

    logger.debug("Loading the featurized dataset...")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["DATE"])
    
    final_predictions_df = pd.DataFrame()

    # Check the ticker_symbol parameter
    if ticker_symbol:
        ticker_symbol = ticker_symbol.upper() + '.SA'
        stock_df_feat_all = stock_df_feat_all[stock_df_feat_all["STOCK"] == ticker_symbol]

    # Check the model_type parameter 
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from: {available_models}")
    
    elif model_type:
        available_models = [model_type]

    for ticker_symbol in stock_df_feat_all["STOCK"].unique():
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["STOCK"] == ticker_symbol].copy()

        for model_type in available_models:
            logger.info(f"Performing inferece for ticker symbol [{ticker_symbol}] using model [{model_type}]...")
 
            current_prod_model = load_production_model_sklearn(model_type, ticker_symbol)

            logger.debug("Creating the future dataframe...")
            future_df = make_future_df(FORECAST_HORIZON, stock_df_feat, features_list)

            logger.debug("Predicting...")
            predictions_df = make_predict(
                model=current_prod_model,
                forecast_horizon=FORECAST_HORIZON-2,
                future_df=future_df,
                past_target_values=list(stock_df_feat[TARGET_NAME].values)
            )

            predictions_df["STOCK"] = ticker_symbol
            predictions_df['MODEL_TYPE'] = model_type
            
            final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)

    if write_to_table:
        logger.debug("Writing the predictions to database...")
        final_predictions_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), index=False)
        logger.debug("Predictions written successfully!")

    return final_predictions_df

    


# Execute the whole pipeline
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform Out-of-Sample Tree-based models Inference.")

    parser.add_argument(
        "-mt", "--model_type",
        type=str,
        choices=["xgb", "et"],
        help="Model name use for inference (xgb, et) (optional, defaults to all)."
    )
    parser.add_argument(
        "-ts", "--ticker_symbol",
        type=str,
        help="""Ticker Symbol for inference. (optional, defaults to all).
        Example: bova11 -> BOVA11.SA | petr4 -> PETR4.SA"""
    )
    parser.add_argument(
        "-w", "--write_to_table",
        action="store_false",
        help="Enable Writing the OFS forecasts to table."
    )
    args = parser.parse_args()

    logger.info("Starting the Inference pipeline...")
    inference_pipeline(args.model_type, args.ticker_symbol, args.write_to_table)
    logger.info("Inference Pipeline completed successfully!")