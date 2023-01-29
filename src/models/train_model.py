# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/home/michel/Projects/mlops_practice/src")
from utils import *

# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

STOCK_NAME = str(input("Which stock do you want to track? "))
make_dataset(STOCK_NAME, PERIOD, INTERVAL)

# load the raw dataset
stock_df = pd.read_csv("./data/raw/raw_stock_prices.csv", parse_dates=True)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# perform featurization
features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]
stock_df_feat = build_features(stock_df, features_list)

# train test split
X_train, X_test, y_train, y_test = ts_train_test_split(stock_df_feat, model_config["TARGET_NAME"], model_config["FORECAST_HORIZON"])

def main():
    predictions_df, X_train_new, y_train_new = make_out_of_sample_predictions(
            X=pd.concat([X_train, X_test], axis=0),
            y=pd.concat([y_train, y_test], axis=0),
            forecast_horizon=model_config["FORECAST_HORIZON"]
        )

# Execute the whole pipeline
if __name__ == "__main__":
    main()

    logger.info("Pipeline was sucessful!")
