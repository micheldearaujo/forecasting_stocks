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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb

# front-end
import streamlit as st
import altair as alt
import plotly.express as px

# MLOps
#import mlflow


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
logger = logging.getLogger("describe")
logger.setLevel(logging.INFO)