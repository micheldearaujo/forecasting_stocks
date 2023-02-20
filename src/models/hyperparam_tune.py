import sys
sys.path.insert(0,'.')

from src.utils import *


def objective(search_space: dict):
    """Objective function to be minimized. Here you define
    the way you want to calculate your loss.

    Args:
        search_space (dict):  Dictionary of hyperparameters to be tuned.

    Returns:
        _type_:  Dictionary of the loss and status.
    """
    
    # create model instance
    model = xgb.XGBRegressor(
        subsample= xgboost_fixed_model_config['SUBSAMPLE'],
        seed = xgboost_fixed_model_config['SEED'],
        **search_space
    )

    # fit the model
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds = 50,
        eval_metric = ['mae'],
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=0,
    )
    
    # evaluate the model
    logger.debug("Evaluating the Hyperopt model...")
    model_mape, model_rmse, model_mae = stepwise_forecasting(model, X_val, y_val, model_config["FORECAST_HORIZON"])
    
    return {'loss': model_mae, 'status': STATUS_OK}


def optimize_model_params(objective_function, search_space: dict):
    """Function that perform hyperparameter tuning using Hyperopt and returns the best parameters.

    Args:
        objective_function (function):  Python function hat returns the loss.
        search_space (dict):  Dictionary of hyperparameters to be tuned.

    Returns:
        xgboost_best_params (dict):  Dictionary of the loss and status.
    """

    # define the algo
    algorithm = tpe.suggest

    logger.debug("Running the Fmin()")
    eval_params = fmin(
        fn=objective,
        space=search_space,
        algo=algorithm,
        max_evals=30,
        verbose=False,
        show_progressbar=True,
    )
    
    logger.debug("Extracting the best params...")
    xgboost_best_params = space_eval(search_space, eval_params)
    
    #{k: v for k, v in best_params.items() if k in xgboost_hyperparameter_config.keys()}
    
    return xgboost_best_params

# Execute the whole pipeline

if __name__ == "__main__":

    logger.info("Starting the Cross Validation pipeline..")

    logger.debug("Loading the featurized dataset..")
    stock_df_feat = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

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

    # call the optimization function
    xgboost_best_params = optimize_model_params(objective, xgboost_hyperparameter_config)

    # now train the model with the best params just to save them
    logger.debug("Training the model with the best params...")
    xgboost_model = xgb.XGBRegressor(
        **xgboost_best_params
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

