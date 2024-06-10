# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import os
import logging.config
import yaml

from src.utils import *


with open("src/configuration/project_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())


with open("src/configuration/logging_config.yaml", 'r') as f:  

    logging_config = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_config)


logger = logging.getLogger(__name__)

### Versao 2: Calcular os lags e médias móveis baseado em uma lista de inteiros
def create_lag_features(df: pd.DataFrame, lag_values: list, target_column: str = "CLOSE") -> pd.DataFrame:
    """
    Creates lag features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        lag_values (list): A list of integers specifying the lag values (e.g., [1, 2, 5] for 1-day, 2-day, and 5-day lags).
        target_column (str, optional): The name of the column to create lag features for (default: "CLOSE").

    Returns:
        pd.DataFrame: The input DataFrame with additional lag features.
    """

    for lag in lag_values:
        df[f"{target_column}_LAG_{lag}"] = df[target_column].shift(lag)
    return df


def create_moving_average_features(df: pd.DataFrame, ma_values: list, target_column: str = "CLOSE") -> pd.DataFrame:
    """
    Creates moving average features for the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target column.
        ma_values (list): A list of integers specifying the window sizes for the moving averages (e.g., [5, 10] for 5-day and 10-day moving averages).
        target_column (str, optional): The name of the column to create moving average features for (default: "CLOSE").

    Returns:
        pd.DataFrame: The input DataFrame with additional moving average features.
    """

    for ma in ma_values:
        df[f"{target_column}_MA_{ma}"] = df[target_column].rolling(ma, closed='left').mean()
    return df


def create_date_features(df: pd.DataFrame, date_column: str = "DATE") -> pd.DataFrame:
    """Creates date-based features from the specified date column."""

    df['DAY_OF_MONTH'] = df[date_column].dt.day
    df['MONTH'] = df[date_column].dt.month
    df['QUARTER'] = df[date_column].dt.quarter
    df['DAY_OF_WEEK'] = df[date_column].dt.weekday
    df['WEEK_OF_MONTH'] = (df['DAY_OF_MONTH'] - 1) // 7 + 1
    df['YEAR'] = df[date_column].dt.year

    return df


def build_features(raw_df: pd.DataFrame, features_list: list, save: bool = True) -> pd.DataFrame:
    """
    Creates features for a machine learning dataset from raw stock data.

    Args:
        raw_df: Raw Pandas DataFrame containing stock data.
        features_list: List of feature names to create.
        save: Whether to save the processed data to a CSV file (default: True).

    Returns:
        Pandas DataFrame with the new features.
    """
    logger.debug("Building features...")
    TARGET_NAME = config['model_config']['TARGET_NAME']

    final_df_featurized = pd.DataFrame()

    for stock_name in raw_df["STOCK"].unique():
        logger.debug("Building features for stock %s..."%stock_name)

        stock_df = raw_df[raw_df["STOCK"] == stock_name].copy()
        
        # Date-based Features
        stock_df = create_date_features(df=stock_df)

        # Round Close Prices (if present)
        if "CLOSE" in stock_df.columns:
            stock_df["CLOSE"] = stock_df["CLOSE"].round(2)

        stock_df = create_lag_features(stock_df, lag_values=[int(f.split("_")[-1]) for f in features_list if "LAG" in f])
        stock_df = create_moving_average_features(stock_df, ma_values=[int(f.split("_")[-1]) for f in features_list if "MA" in f])

        stock_df.dropna(inplace=True)

        final_df_featurized = pd.concat([final_df_featurized, stock_df], axis=0)

    final_df_featurized.columns = final_df_featurized.columns.str.upper() 
    final_df_featurized = final_df_featurized.reindex(columns=["DATE", "STOCK", TARGET_NAME, *features_list])

    if save:
        write_dataset_to_file(final_df_featurized, PROCESSED_DATA_PATH)

    logger.debug("Features built successfully!")
    logger.debug(f"\n{final_df_featurized.tail()}")
    logger.debug(f"Dataset shape: {final_df_featurized.shape}.")
    logger.debug(f"Amount of ticker symbols: {final_df_featurized['STOCK'].nunique()}.")

    return final_df_featurized



if __name__ == '__main__':

    logger.debug("Loading the raw dataset to featurize it...")
    stock_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), parse_dates=["DATE"])

    logger.info("Featurizing the dataset...")
    features_list = config['features_list']
    stock_df_feat = build_features(stock_df, features_list)
    
    logger.info("Finished featurizing the dataset!")