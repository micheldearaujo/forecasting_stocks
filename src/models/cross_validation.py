# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

# Execute the whole pipeline

if __name__ == "__main__":

    #STOCK_NAME = str(input("Which stock do you want to track? "))
    STOCK_NAME = 'BOVA11.SA'
    logger.info("Starting the Cross Validation pipeline..")

    # download the dataset and as raw
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    param_grid = {
        "n_estimators": [40, 100, 200, 300, 400],
        "max_depth": [3, 5, 7, 9, 12],
        "learning_rate": [0.2, 0.3, 0.1, 0.01, 0.001],
        "subsample": [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "gamma": [0, 0.25, 0.5, 1.0],
        "reg_alpha": [0, 0.25, 0.5, 1.0],
        "reg_lambda": [0, 0.25, 0.5, 1.0],
    }

    logger.info("Start Cross Validation..")
    # validate the  model on full historical data
    cross_val_data = stock_df_feat[stock_df_feat.Date < stock_df_feat.Date.max() - dt.timedelta(days=model_config["FORECAST_HORIZON"])]

    # best_model, best_params, best_score = time_series_grid_search_xgb(
    #     data=cross_val_data.drop([model_config["TARGET_NAME"], "Date"], axis=1),
    #     target=cross_val_data[model_config["TARGET_NAME"]],
    #     param_grid=param_grid,
    #     n_splits=3,
    #     random_state=0,
    #     stock_name=STOCK_NAME
    # )

    #print(best_params, best_score)

    predictions_df = validade_model_one_shot(
        X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
        y=stock_df_feat[model_config["TARGET_NAME"]],
        forecast_horizon=model_config['FORECAST_HORIZON'],
        stock_name=STOCK_NAME
    )

    logger.info("Cross Validation Pipeline was sucessful!")