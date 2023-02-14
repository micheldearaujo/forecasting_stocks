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
    logger.info("Starting the Validation pipeline..")

    # download the dataset
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    predictions_df = validade_model_one_shot(
        X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
        y=stock_df_feat[model_config["TARGET_NAME"]],
        forecast_horizon=model_config['FORECAST_HORIZON'],
        stock_name=STOCK_NAME
    )

    #cd_pipeline()


    logger.info("Validation Pipeline was sucessful!")