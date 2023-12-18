# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from src.utils import *
from src.config import xgboost_model_config
from src.models.train_model import extract_learning_curves
from src.models.model_utils import cd_pipeline

logger = logging.getLogger("model-testing")
logger.setLevel(logging.INFO)

def test_model_one_shot(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, stock_name: str) -> pd.DataFrame:
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
    predictions = []
    actuals = []
    dates = []
    X_testing_df = pd.DataFrame()
    
    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon+4, :]
    y_train = y.iloc[:-forecast_horizon+4]

    final_y = y_train.copy()
    # # start the mlflow tracking
    mlflow.set_experiment(experiment_name="model-testing")
    with mlflow.start_run(run_name=f"xgboost_{stock_name}") as run:

        # fit the model again with the best parameters
        xgboost_model = xgb.XGBRegressor(
            eval_metric=["rmse", "logloss"],
            **xgboost_model_config
        )

        logger.debug("Fitting the model...")
        xgboost_model.fit(
            X_train.drop("Date", axis=1),
            y_train,
            eval_set=[(X_train.drop("Date", axis=1), y_train)],
            verbose=0
        )

        # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
        # After forecasting the next step, we need to update the "lag" features with the last forecasted
        # value
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


def model_testing_pipeline():

    logger.debug("Loading the featurized dataset..")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    # iterate over the stocks
    validation_report_df = pd.DataFrame()

    for stock_name in stock_df_feat_all["Stock"].unique():

        logger.info("Testing the model for the stock: %s..."%stock_name)
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].copy().drop("Stock", axis=1)
        
        predictions_df = test_model_one_shot(
            X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
            y=stock_df_feat[model_config["TARGET_NAME"]],
            forecast_horizon=model_config['FORECAST_HORIZON'],
            stock_name=stock_name
        )

        predictions_df["Stock"] = stock_name
        predictions_df["Training_Date"] = dt.datetime.today().date()

        validation_report_df = pd.concat([validation_report_df, predictions_df], axis=0)
    
    logger.debug("Writing the testing results dataframe...")
    validation_report_df = validation_report_df.rename(columns={"Forecast": "Price"})
    validation_report_df["Class"] = "Testing"

    validation_report_df.to_csv(os.path.join(OUTPUT_DATA_PATH, 'validation_stock_prices.csv'), index=False)

# Execute the whole pipeline

if __name__ == "__main__":
    logger.info("Starting the Model Testing pipeline...")

    model_testing_pipeline()
    
    logger.info("Model Testing Pipeline was sucessful!")