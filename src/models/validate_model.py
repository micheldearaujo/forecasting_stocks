# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *


# Execute the whole pipeline

if __name__ == "__main__":

    logger.debug("Loading the featurized dataset..")
    stock_df_feat = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    logger.info("Starting the Model Evaluation pipeline..")
    predictions_df = validade_model_one_shot(
        X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
        y=stock_df_feat[model_config["TARGET_NAME"]],
        forecast_horizon=model_config['FORECAST_HORIZON'],
        stock_name=STOCK_NAME
    )

    logger.info("Model Evaluation Pipeline was sucessful!")