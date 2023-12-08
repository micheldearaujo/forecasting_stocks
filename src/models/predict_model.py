import sys
sys.path.insert(0,'.')

import re

from src.utils import *
from src.features.build_features import build_features
from src.models.model_utils import (
    make_predict
)
from src.config import features_list

logger = logging.getLogger("Inference_Pipeline")
logger.setLevel(logging.DEBUG)

def predict_pipeline():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.

    Parameters:
        None
    Returns:
        None
    """

    client = MlflowClient()

    logger.debug("Loading the featurized dataset...")
    # load the featurized dataset
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    
    # create empty dataset to store all the predictions
    final_predictions_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():

        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy()

        logger.debug(f"Loading the production model for stock {stock_name}...")

        # Need to load the current prod inference model now, to archive it
        models_versions = []

        for mv in client.search_model_versions("name='{}_{}'".format(model_config[f'REGISTER_MODEL_NAME_INF'], stock_name)):
            models_versions.append(dict(mv))

        # Check if there is production model
        try:
            logger.debug("Previous model version found. Loading it...") 
    
            current_prod_inf_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]
            current_model_path = current_prod_inf_model['source']
            current_model_path = re.search(r'mlruns.*/model', current_model_path)
            current_model_path = current_model_path.group()

            # TODO: Corrigir esse xgboost_model aqui, pois pode ser qualquer modelo
            # Sugestão: 'xgboost_model_' vira "best_model"
            current_model_path = current_model_path[:-5] + 'xgboost_model_' + stock_name
            logger.debug(f"Loading model from path: \n{current_model_path}")
            current_prod_inf_model = mlflow.xgboost.load_model(model_uri='./'+current_model_path)


        except IndexError:

            logger.warning("NO PRODUCTION MODEL FOUND. STOPPING THE PIPELINE! ")
            break

        # Create the future dataframe using the make_future_df function
        logger.debug("Creating the future dataframe...")
        future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)

        # drop the stock column
        future_df = future_df.drop("Stock", axis=1)
        
        # Make predictions using the future dataframe and specified forecast horizon
        logger.debug("Making predictions...")
        
        predictions_df = make_predict(
            model=current_prod_inf_model,
            forecast_horizon=model_config["FORECAST_HORIZON"]-4,
            future_df=future_df
        )

        # add the stock name to the predictions dataframe
        predictions_df["Stock"] = stock_name
        
        # concat the predictions to the final predictions dataframe
        final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)

    # write the predictions to a csv file
    logger.debug("Writing the predictions to database...")

    final_predictions_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), index=False)

    logger.debug("Predictions written successfully!")


# Execute the whole pipeline
if __name__ == "__main__":

    logger.info("Starting the Inference pipeline...\n")

    predict_pipeline()

    logger.info("Inference Pipeline was successful!\n")

