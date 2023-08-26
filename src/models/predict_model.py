import sys
sys.path.insert(0,'.')

from src.utils import *
from src.features.build_features import build_features
from src.models.model_utils import (
    make_predict
)

logger = logging.getLogger("Inference_Pipeline")
logger.setLevel(logging.INFO)

features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]

def predict_pipeline():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.


    ----- THIS FUNCTION IS NOT USING THE MLFLOW MODEL. IT IS USING THE JOBLIB ONE. FIX THAT !! ---

    Parameters:
        None
    Returns:
        None
    """

    client = MlflowClient()

    logger.debug("Loading the featurized dataset..")
    # load the featurized dataset
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    
    # create empty dataset to store all the predictions
    final_predictions_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():

        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy()

        logger.debug(f"\nLoading the production model for stock {stock_name}...")

        # -------------------- FIX THE MLFLOW MODEL LOADING --------------------
        # create empty list to store the model versions
        models_versions = []
        for mv in client.search_model_versions("name='{}_{}'".format(model_config['REGISTER_MODEL_NAME_INF'], stock_name)):
            models_versions.append(dict(mv))

        # load the production model
        #current_prod_model_info = [x for x in models_versions if x['current_stage'] == 'Production'][0]
        #current_prod_model_uri = f"./mlruns/0/{current_prod_model_info['run_id']}/artifacts/{model_config['MODEL_NAME']}_{stock_name}"
        #xgboost_model = mlflow.xgboost.load_model(model_uri=current_prod_model_uri)
        # -------------------------------------------------------------------------
        
        xgboost_model = load(f"./models/{stock_name}_{dt.datetime.today().date()}.joblib")

        # Create the future dataframe using the make_future_df function
        logger.debug("Creating the future dataframe...")
        future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)

        # drop the stock column
        future_df = future_df.drop("Stock", axis=1)
        
        # Make predictions using the future dataframe and specified forecast horizon
        logger.debug("Making predictions...")
        
        predictions_df = make_predict(
            model=xgboost_model,
            forecast_horizon=model_config["FORECAST_HORIZON"]-4,
            future_df=future_df
        )

        # add the stock name to the predictions dataframe
        predictions_df["Stock"] = stock_name
        
        # concat the predictions to the final predictions dataframe
        final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)

    # write the predictions to a csv file
    logger.debug("Writing the predictions to a csv file...")

    final_predictions_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), index=False)

    logger.debug("Predictions written sucessfully!")


# Execute the whole pipeline
if __name__ == "__main__":

    logger.info("Starting the Inference pipeline...\n")

    predict_pipeline()

    logger.info("\nInference Pipeline was sucessful!\n")

