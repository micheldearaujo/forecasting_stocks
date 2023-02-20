# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
from src.models.train_model import extract_learning_curves

def validade_model_one_shot(X: pd.DataFrame, y: pd.Series, forecast_horizon: int, stock_name: str) -> pd.DataFrame:
    """
    Make predictions for the next `forecast_horizon` days using a XGBoost model.
    This model is validated using One Shot Training, it means that we train the model
    once, and them perform the `forecast_horizon` predictions only loading the mdoel.
    
    Parameters:
        X (pandas dataframe): The input data
        y (pandas dataframe): The target data
        forecast_horizon (int): Number of days to forecast
        
    Returns:
        pred_df: Pandas DataFrame with the forecasted values
    """

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []
    
    # get the one-shot training set
    X_train = X.iloc[:-forecast_horizon, :]
    y_train = y.iloc[:-forecast_horizon]
    
    # load the best model
    xgboost_model = load(f"./models/{stock_name}_params.joblib")
    # get the best parameters
    parameters = xgboost_model.get_xgb_params()
    parameters.pop("eval_metric")

    # start the mlflow tracking
    with mlflow.start_run(run_name="model_validation") as run:

        # fit the model again with the best parameters
        xgboost_model = xgb.XGBRegressor(
            **parameters
        )

        # train the model
        xgboost_model.fit(
            X_train.drop("Date", axis=1),
            y_train,
            eval_set=[(X_train.drop("Date", axis=1), y_train)],
            eval_metric=["rmse", "logloss"],
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
            # because the following ones will use the last predicted value as true value
            # so we simulate the process of predicting out-of-sample
            if len(predictions) != 0:
                
                # we need to update the X_test["Close_lag_1"] value, because
                # it should be equal to the last prediction (the "yesterday" value)
                X_test.iat[0, -1] = predictions[-1]            

            else:
                pass

            # make prediction
            prediction = xgboost_model.predict(X_test.drop("Date", axis=1))

            # store the results
            predictions.append(prediction[0])
            actuals.append(y_test.values[0])
            dates.append(X_test["Date"].max())

        # Calculate the resulting metric
        model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
        model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)

        pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["Date", 'Actual', 'Forecast'])
        pred_df["Forecast"] = pred_df["Forecast"].astype("float64")

        # Plotting the Validation Results
        fig = visualize_validation_results(pred_df, model_mape, model_rmse, stock_name)
      
        # Plotting the Learning Results
        fig2 = extract_learning_curves(xgboost_model)

        # ---- logging ----
        logger.debug("Logging the results to MLFlow")
        # log the parameters
        mlflow.log_params(parameters)

        # log the metrics
        mlflow.log_metric("MAPE", model_mape)
        mlflow.log_metric("RMSE", model_rmse)

        # log the figure
        mlflow.log_figure(fig, "validation_results.png")
        mlflow.log_figure(fig2, "learning_curves.png")

        # get model signature
        model_signature = infer_signature(X_train, pd.DataFrame(y_train))

        # log the model to mlflow
        mlflow.xgboost.log_model(
            xgb_model=xgboost_model,
            artifact_path="xgboost_model",
            input_example=X_train.head(),
            signature=model_signature
        )

        # execute the CD pipeline
        cd_pipeline(run.info, y_train, pred_df, model_mape)

    
    return pred_df


def model_validation_pipeline():

    logger.debug("Loading the featurized dataset..")
    stock_df_feat = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    
    predictions_df = validade_model_one_shot(
        X=stock_df_feat.drop([model_config["TARGET_NAME"]], axis=1),
        y=stock_df_feat[model_config["TARGET_NAME"]],
        forecast_horizon=model_config['FORECAST_HORIZON'],
        stock_name=STOCK_NAME
    )

# Execute the whole pipeline

if __name__ == "__main__":
    logger.info("Starting the Model Evaluation pipeline..")

    model_validation_pipeline()
    
    logger.info("Model Evaluation Pipeline was sucessful!")