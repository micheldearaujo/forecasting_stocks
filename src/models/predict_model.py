import sys
sys.path.insert(0,'.')

from src.utils import *



def main():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.

    Parameters:
        None
    Returns:
        None
    """

    # make the dataset
    PERIOD = '800d'
    INTERVAL = '1d'
    STOCK_NAME = 'BOVA11.SA'  #str(input("Which stock do you want to track? "))

    features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]
    client = MlflowClient()
    models_versions = []

    logger.debug("Loading the featurized dataset..")
    # load the featurized dataset
    stock_df_feat = pd.read_csv("./data/processed/processed_stock_prices.csv", parse_dates=["Date"])

    logger.debug("Loading the production model...")
    # load the production model
    for mv in client.search_model_versions("name='{}'".format(model_config['REGISTER_MODEL_NAME_INF'])):
        models_versions.append(dict(mv))

    current_prod_model_info = [x for x in models_versions if x['current_stage'] == 'Production'][0]
    current_prod_model_uri = f"./mlruns/0/{current_prod_model_info['run_id']}/artifacts/xgboost_model"
    xgboost_model = mlflow.xgboost.load_model(model_uri=current_prod_model_uri)
    logger.debug("Production model loaded sucessfully!")
    
    logger.debug("Creating the future dataframe...")
    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)
    logger.debug("Future dataframe created sucessfully!")
    
    # Make predictions using the future dataframe and specified forecast horizon
    logger.debug("Making predictions...")
    predictions_df = make_predict(
        model=xgboost_model,
        forecast_horizon=model_config["FORECAST_HORIZON"]-4,
        future_df=future_df
    )
    logger.debug("Predictions made sucessfully!")

    # write the predictions to a csv file
    logger.debug("Writing the predictions to a csv file...")

    predictions_df.to_csv("./data/output/output_stock_prices.csv", index=False)

    logger.debug("Predictions written sucessfully!")



# Execute the whole pipeline
if __name__ == "__main__":

    logger.info("Starting the Inference pipeline..\n")

    main()

    logger.info("Inference Pipeline was sucessful!\n")

