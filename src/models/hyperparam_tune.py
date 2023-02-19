import sys
sys.path.insert(0,'.')

from src.utils import *

# TODO: Create the Hyperparameter Optimization Pipeline (On hold)

# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

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
    
    model_mape, model_rmse, model_mae = stepwise_forecasting(model, X_val, y_val, model_config["FORECAST_HORIZON"])
    
    return {'loss': model_mae, 'status': STATUS_OK}


# Execute the whole pipeline

if __name__ == "__main__":

    #STOCK_NAME = str(input("Which stock do you want to track? "))
    STOCK_NAME = 'BOVA11.SA'
    logger.info("Starting the Cross Validation pipeline..")

    # TODO: Stop downloading the dataset every time, just load it
    # download the dataset
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    # perform cross-validation until the last forecast horizon
    cross_val_data = stock_df_feat[stock_df_feat.Date < stock_df_feat.Date.max() - dt.timedelta(days=model_config["FORECAST_HORIZON"])]

    # training and testing test
    #X = cross_val_data.drop([model_config["TARGET_NAME"], "Date"], axis=1)
    #y = cross_val_data[model_config["TARGET_NAME"]]
    logger.debug("Splitting the dataset into train and test sets..")
    X_train, X_test, y_train, y_test = ts_train_test_split(cross_val_data, model_config["TARGET_NAME"], test_size=model_config["FORECAST_HORIZON"]-4)
    X_train = X_train.drop("Date", axis=1)
    X_test = X_test.drop("Date", axis=1)
    
    logger.debug("Getting the validation set")
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
        max_evals=10,
    )
    
    logger.debug("Extracting the best params")
    xgboost_best_param_names = space_eval(search_space, best_params)
    print(xgboost_best_param_names)

    # now train the model with the best params just to save them
    xgboost_model = xgb.XGBRegressor(
        **xgboost_best_param_names
    )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
    )

    # save it
    dump(xgboost_model, f"./models/{STOCK_NAME}_params.joblib")
    logger.info("Cross Validation Pipeline was sucessful!")

