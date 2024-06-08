# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
import warnings
import yaml
import argparse
import logging

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

with open("src/configuration/logging_config.yaml", 'r') as f:  

    loggin_config = yaml.safe_load(f.read())
    logging.config.dictConfig(loggin_config)

with open("src/configuration/hyperparams.yaml", 'r') as f:  

    hyperparams = yaml.safe_load(f.read())

with open("src/configuration/project_config.yaml", 'r') as f:  

    model_config = yaml.safe_load(f.read())['model_config']

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
        

def train_model(X_train, y_train, model_type, ticker_symbol, tune_params):
    """Trains a tree-based regression model."""

    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH+f'/{model_type}', exist_ok=True)
    base_params = hyperparams['BASE_PARAMS'][model_type]

    if tune_params:
        best_params = tune_params_gridsearch(X_train, y_train, model_type, ticker_symbol)
        base_params.update(best_params)

    if model_type == 'xgb':
        xgb.set_config(verbosity=1)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(base_params, dtrain, num_boost_round=200)
        model.save_model(f"{MODELS_PATH}/{model_type}/Model_{ticker_symbol}.xgb")
        return model

    elif model_type == 'et':
        model = ExtraTreesRegressor(**base_params).fit(X_train, y_train)
        dump(model, f"{MODELS_PATH}/{model_type}/Model_{ticker_symbol}.json")
        return model
    

def predict(model, X_test, model_type):
    """Realiza previsões com o modelo XGBoost treinado."""

    if model_type == 'xgb':
        dtest = xgb.DMatrix(X_test)
        return model.predict(dtest)

    elif model_type == 'et':
        return model.predict(X_test)

    elif model_type == 'nbeats':
        return model.predict(X_test)


def tune_params_gridsearch(X: pd.DataFrame, y: pd.Series, model_type:str, ticker_symbol: str, n_splits=5):
    """
    Performs time series hyperparameter tuning on a model using grid search.
    
    Args:
        X (pd.DataFrame): The input feature data
        y (pd.Series): The target values
        param_grid (dict): Dictionary of hyperparameters to search over
        n_splits (int): Number of folds for cross-validation (default: 5)
        random_state (int): Seed for the random number generator (default: 0)
    
    Returns:
        tuple: A tuple containing the following elements:
            best_model (xgb.XGBRegressor): The best XGBoost model found by the grid search
            best_params (dict): The best hyperparameters found by the grid search
    """

    logger.info(f"Performing hyperparameter tuning for {ticker_symbol} using {model_type}...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = xgb.XGBRegressor() if model_type == "xgb" else eval(model_type + "Regressor")

    param_distributions = hyperparams['PARAM_SPACES'][model_type]

    grid_search = GridSearchCV(
        model,
        param_grid=param_distributions,
        cv=tscv,
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        verbose=1
    ).fit(X, y)
    
    best_params = grid_search.best_params_
    logger.info(f"Best parameters found: {best_params}")

    return best_params


def model_training_pipeline(tune_params=False):


    logger.debug("Loading the featurized dataset..")
    all_ticker_symbols_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    for ticker_symbol in all_ticker_symbols_df["Stock"].unique():
        logger.info("Creating training dataset for Ticker Symbol [%s...]"%ticker_symbol)

        ticker_df_feat = all_ticker_symbols_df[all_ticker_symbols_df["Stock"] == ticker_symbol].drop("Stock", axis=1).copy()

        X_train=ticker_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1)
        y_train=ticker_df_feat[model_config["TARGET_NAME"]]

        logger.debug("Training the models..")
        xgb_model = train_model(X_train, y_train, 'xgb', ticker_symbol, tune_params)
        # et_model = train_model(X_train, y_train, 'et', ticker_symbol, tune_params)

        logger.debug("Plotting the learning curves..")
        # learning_curves_fig , feat_importance_fig = extract_learning_curves(xgboost_model)



# Execute the whole pipeline
if __name__ == "__main__":
    logger.info("Starting the training pipeline...")

    model_training_pipeline(True)

    logger.info("Training Pipeline was sucessful!\n")

