# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import date
import pandas_datareader as web
import yfinance as yfin
import datetime as dt
import sys
import os
import logging
from joblib import load, dump

# Time Series Libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  #Autocorrelação (MA), Autocorrelatcao parcial (AR)ve
from pmdarima.arima.utils import ndiffs 

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from hyperopt import fmin, tpe, Trials, hp, SparkTrials, space_eval, STATUS_OK, rand, Trials

# front-end
import streamlit as st
import altair as alt
import plotly.express as px

# MLOps
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient




plt.style.use("fivethirtyeight")

# Define dates to start and end
initial_stock_date = dt.datetime.now().date() - dt.timedelta(days=3*365)
final_stock_date = dt.datetime.now().date()

model_config = {
    "TEST_SIZE": 0.2,
    "TARGET_NAME": "Close",
    "VALIDATION_METRIC": "MAPE",
    "OPTIMIZATION_METRIC": "MSE",
    "FORECAST_HORIZON": 14,
    "REGISTER_MODEL_NAME_VAL": "Stock_Predictor_Validation",
    "REGISTER_MODEL_NAME_INF": "Stock_Predictor_Inference"
}

features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]

# Define a ação para procurar
PERIOD = '800d'
INTERVAL = '1d'
STOCK_NAME = 'BOVA11.SA'

# Configura o logging
log_format = "[%(name)s][%(levelname)-6s] %(message)s"
logging.basicConfig(format=log_format)
logger = logging.getLogger("Status")
logger.setLevel(logging.DEBUG)


# paths
ROOT_DATA_PATH = "./data"
RAW_DATA_PATH = os.path.join(ROOT_DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(ROOT_DATA_PATH, "processed")
EXTERNAL_DATA_PATH = os.path.join(ROOT_DATA_PATH, "external")
INTERIM_DATA_PATH = os.path.join(ROOT_DATA_PATH, "interim")

MODELS_PATH = "./models"

param_grid = {
    "n_estimators": [40, 100, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.2, 0.3, 0.1, 0.01, 0.001],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [1.0],
    "gamma": [0.1, 0.25, 0.5, 1.0],
    "reg_alpha": [0, 0.25, 0.5, 1.0],
    "reg_lambda": [0, 0.25, 0.5, 1.0],
}

xgboost_model_config = {
    'LEARNING_RATE': 0.01,
    'MAX_DEPTH': 100,
    'MIN_DATA': 100,
    'N_ESTIMATORS': 1000,
    'REG_LAMBDA': 100,
    'SCALE_POS_WEIGHT': 10,
    'SEED': 42,
    'SUBSAMPLE': 0.9,
    'COLSAMPLE_BYTREE': 0.9,
    'NUM_BOOST_ROUNDS': 200,
    'GAMMA': 0.01
}

xgboost_fixed_model_config = {
    'SEED': 42,
    'SUBSAMPLE': 1.0
}
xgboost_hyperparameter_config = {
    'max_depth': hp.choice('max_depth', [4, 9, 11, 30]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.08 ,0.1, 0.5, 1.0]),
    'gamma': hp.choice('gamma', [0.01, 0.08, 0.1, 1.0]),
    'reg_lambda': hp.choice('reg_lambda', [1, 10, 30, 100]),
    'n_estimators': hp.choice('n_estimators', [40, 200, 300, 1000]),
    'scale_pos_weight': hp.choice('scale_pos_weight', [1, 2, 3, 4, 10, 15]),
    'colsample_bytree': hp.choice('colsample_bytree', [1.0]),
}