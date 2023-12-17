import sys
sys.path.insert(0,'.')

import re
from src.utils import *
from src.features.build_features import build_features
from src.models.model_utils import (
    make_predict
)
from src.config import features_list

logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG)

def load_production_model(logger, model_config, stock_name):

    client = MlflowClient()

    logger.info(f"Searching for production models for stock {stock_name}...")
    models_versions = []

    for mv in client.search_model_versions("name='{}_{}'".format(model_config[f'REGISTER_MODEL_NAME_INF'], stock_name)):
        models_versions.append(dict(mv))

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
        current_prod_model = mlflow.xgboost.load_model(model_uri='./'+current_model_path)

        return current_prod_model

    except IndexError:
        logger.warning("NO PRODUCTION MODEL FOUND. STOPPING THE PIPELINE!")
        return None
    

def make_predict(model, forecast_horizon: int, future_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for the past `forecast_horizon` days using a XGBoost model.
    This model is validated using One Shot Training, it means that we train the model
    once, and them perform the `forecast_horizon` predictions only loading the mdoel.
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas Series): The target data
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        pred_df: Pandas DataFrame with the forecasted values
    """

    # Create empty list for storing each prediction
    future_df_feat = future_df.copy()

    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()

    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # After forecasting the next step, we need to update the "lag" features with the last forecasted
    # value
    # TODO: Continuar daqui a revisão e adequação do predict future com média móvel
    for day in range(forecast_horizon-4, 0, -1):
        
        if day != 1:
            # the testing set will be the next day after the training and we use the complete dataset
            X_test = X.iloc[-day:-day+1,:]
            y_test = y.iloc[-day:-day+1]

        else:
            # need to change the syntax for the last day (for -1:-2 will not work)
            X_test = X.iloc[-day:,:]
            y_test = y.iloc[-day:]

        # only the first iteration will use the true value of Close_lag_1
        # because the following ones will use the last predicted value
        # so we simulate the process of predicting out-of-sample
        if len(predictions) != 0:
            
            # ------------ UPDATE THE "LAG_1" FEATURE USING THE LAST PREDICTED VALUE ------------
            # we need to update the X_test["Close_lag_1"] value, because
            # it should be equal to the last prediction (the "yesterday" value)
            X_test.iat[0, -1] = predictions[-1]

            # ------------ UPDATE THE "MA_7" FEATURE, USING THE LAST 7 "Target" VALUES ------------
            # re-calculating the Moving Average.
            # we need to extract he last closing prices before today
            # last_closing_princes = final_y.iloc[:-day] # thats -11
            last_closing_princes_ma = final_y.rolling(7).mean()
            last_ma = last_closing_princes_ma.values[-1]
            X_test.iat[0, -2] = last_ma

            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)

        else:
            # we jump the first iteration because we do not need to update anything.
            X_testing_df = pd.concat([X_testing_df, X_test], axis=0)

            pass

        # make prediction
        prediction = xgboost_model.predict(X_test.drop("Date", axis=1))
        final_y = pd.concat([final_y, pd.Series(prediction[0])], axis=0)
        final_y = final_y.reset_index(drop=True)

        # store the results
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["Date"].max())

    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["Date", 'Actual', 'Forecast'])
    pred_df["Forecast"] = pred_df["Forecast"].astype("float64")

    logger.debug("Calculating the evaluation metrics...")
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
    model_mae = round(mean_absolute_error(actuals, predictions), 2)
    model_wape = round((pred_df.Actual - pred_df.Forecast).abs().sum() / pred_df.Actual.sum(), 2)

    pred_df["MAPE"] = model_mape
    pred_df["MAE"] = model_mae
    pred_df["WAPE"] = model_wape
    pred_df["RMSE"] = model_rmse
    pred_df["Model"] = str(type(xgboost_model)).split('.')[-1][:-2]

    # Plotting the Validation Results
    validation_metrics_fig = visualize_validation_results(pred_df, model_mape, model_mae, model_wape, stock_name)

    # Plotting the Learning Results
    learning_curves_fig, feat_imp = extract_learning_curves(xgboost_model, display=False)

    # ---- logging ----
    logger.debug("Logging the results to MLFlow")
    parameters = xgboost_model.get_xgb_params()
    #parameters = xgboost_model_config
    mlflow.log_params(parameters)
    mlflow.log_param("features", list(X_test.columns))

    # log the metrics
    mlflow.log_metric("MAPE", model_mape)
    mlflow.log_metric("RMSE", model_rmse)
    mlflow.log_metric("MAE", model_mae)
    mlflow.log_metric("WAPE", model_wape)

    # log the figure
    mlflow.log_figure(validation_metrics_fig, "validation_results.png")
    mlflow.log_figure(learning_curves_fig, "learning_curves.png")
    mlflow.log_figure(feat_imp, "feature_importance.png")

    # get model signature
    model_signature = infer_signature(X_train, pd.DataFrame(y_train))

    # log the model to mlflow
    mlflow.xgboost.log_model(
        xgb_model=xgboost_model,
        artifact_path=f"xgboost_model_{stock_name}",
        input_example=X_train.head(),
        signature=model_signature
    )

    # execute the CD pipeline
    #cd_pipeline(run.info, y_train, pred_df, model_mape, stock_name)

    
    return pred_df#, X_testing_df



def inference_pipeline():
    """
    Run the inference pipeline for predicting stock prices using the production model.

    This function loads the featurized dataset, searches for the production model for a specific stock,
    and makes predictions for the future timeframe using the loaded model. The predictions are then
    saved to a CSV file.

    Parameters:
        None

    Returns:
        None
    """


    logger.debug("Loading the featurized dataset...")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    
    final_predictions_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():
        logger.info(f"Performing inferece for ticker symbol {stock_name}...")

        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy()
        current_prod_model = load_production_model(logger, model_config, stock_name)

        logger.debug("Creating the future dataframe...")
        future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)
        future_df = future_df.drop("Stock", axis=1)
        
        logger.debug("Predicting...")
        predictions_df, feature_df = make_predict(
            model=current_prod_model,
            forecast_horizon=model_config["FORECAST_HORIZON"]-4,
            future_df=future_df
        )

        predictions_df["Stock"] = stock_name
        
        final_predictions_df = pd.concat([final_predictions_df, predictions_df], axis=0)


    logger.debug("Writing the predictions to database...")
    final_predictions_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), index=False)

    logger.debug("Predictions written successfully!")


# Execute the whole pipeline
if __name__ == "__main__":

    logger.info("Starting the Inference pipeline...")

    inference_pipeline()

    logger.info("Inference Pipeline was successful!\n")

