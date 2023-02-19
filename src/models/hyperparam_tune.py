import sys
sys.path.insert(0,'.')

from src.utils import *


def objective(search_space):
    
    model = xgb.XGBRegressor(
        subsample= xgboost_fixed_model_config['SUBSAMPLE'],
        seed = xgboost_fixed_model_config['SEED'],
        **search_space
    )

    model.fit(
        X_train,
        y_train,
        early_stopping_rounds = 50,
        eval_metric = ['mae'],
        eval_set=[(X_train, y_train), (X_test, y_test)]
    )
    
    logger.debug("Evaluating the Hyperopt model...")
    model_mape, model_rmse, model_mae = stepwise_forecasting(model, X_val, y_val, model_config["FORECAST_HORIZON"])
    
    return {'loss': model_mae, 'status': STATUS_OK}


# Execute the whole pipeline

if __name__ == "__main__":

    logger.info("Starting the Cross Validation pipeline..")

    logger.debug("Loading the featurized dataset..")
    stock_df_feat = pd.read_csv("./data/processed/processed_stock_prices.csv", parse_dates=["Date"])

    # perform cross-validation until the last forecast horizon
    cross_val_data = stock_df_feat[stock_df_feat.Date < stock_df_feat.Date.max() - dt.timedelta(days=model_config["FORECAST_HORIZON"])]

    logger.debug("Splitting the dataset into train and test sets..")
    X_train, X_test, y_train, y_test = ts_train_test_split(cross_val_data, model_config["TARGET_NAME"], test_size=model_config["FORECAST_HORIZON"]-4)
    X_train = X_train.drop("Date", axis=1)
    X_test = X_test.drop("Date", axis=1)
    
    logger.debug("Getting the validation set...")
    # final validation set
    X_val = stock_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1).iloc[-model_config["FORECAST_HORIZON"]+4:, :]
    y_val = stock_df_feat[model_config["TARGET_NAME"]].iloc[-model_config["FORECAST_HORIZON"]+4:]

    # define search space
    search_space = xgboost_hyperparameter_config
    # define algo
    algorithm = tpe.suggest
    # define trials
    #spark_trials = SparkTrials(parallelism=1)
    logger.debug("Running the Fmin()")
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=algorithm,
        max_evals=30,
    )
    
    logger.debug("Extracting the best params...")
    xgboost_best_param_names = space_eval(search_space, best_params)


    # now train the model with the best params just to save them
    logger.debug("Training the model with the best params...")
    xgboost_model = xgb.XGBRegressor(
        **xgboost_best_param_names
    )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
    )

    # save it
    logger.debug("Saving the model with Joblib in order to use the parameters later...")
    dump(xgboost_model, f"./models/{STOCK_NAME}_params.joblib")
    logger.info("Cross Validation Pipeline was sucessful!")

