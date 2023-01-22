# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/michel/Projects/mlops_practice/src')
from utils import *

# Load the raw dataset
stock_df = pd.read_csv('./data/raw/raw_stock_prices.csv', parse_dates=True)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Perform featurization
features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]


stock_df_feat = build_features(stock_df, features_list)

# train test split
X_train, X_test, y_train, y_test = ts_train_test_split(stock_df_feat, model_config["TARGET_NAME"], model_config["FORECAST_HORIZON"])


# Execute the whole pipeline
make_predictions(pd.concat([X_train, X_test], axis=0), pd.concat([y_train, y_test], axis=0), model_config['FORECAST_HORIZON'])