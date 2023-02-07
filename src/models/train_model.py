# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

# Execute the whole pipeline
if __name__ == "__main__":

    STOCK_NAME = str(input("Which stock do you want to track? "))

    logger.info("Starting the training pipeline..")

    # download the dataset and as raw
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    # load the model parameters
    xgboost_model = load(f"./models/{STOCK_NAME}_xgb.joblib")
    parameters = xgboost_model.get_xgb_params()

    # train model on full historical data
    xgboost_model = train_model(
        X_train=stock_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1),
        y_train=stock_df_feat[model_config["TARGET_NAME"]],
        params=parameters
    )

    logger.info("Training Pipeline was sucessful!")
