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
    "REGISTER_MODEL_NAME": "Stock_Predictor"
}

features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]

# Define a ação para procurar
#STOCK_NAME = 'BOVA11.SA'

# Configura o logging
log_format = "[%(name)s][%(levelname)-6s] %(message)s"
logging.basicConfig(format=log_format)
logger = logging.getLogger("Status")
logger.setLevel(logging.INFO)


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
