# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *

# Load the dataset
stock_df = pd.read_csv('./data/raw/raw_stock_prices.csv', parse_dates=True)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Filter the time to include only pos COVID period
#stock_df = stock_df[stock_df['Date'] >= pd.to_datetime('2020-09-01')]


def build_features(raw_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    This function creates the features for the dataset to be consumed by the
    model
    
    :param raw_df: Raw Pandas DataFrame to create the features of
    :param features_list: The list of features to create

    :return: Pandas DataFrame with the new features
    """

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

    # Save the dataset
    stock_df_featurized.to_csv("./data/processed/processed_stock_prices.csv", index=False)

    return stock_df_featurized

