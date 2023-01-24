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


plt.style.use("fivethirtyeight")

# Define dates to start and end
initial_stock_date = dt.datetime.now().date() - dt.timedelta(days=3*365)
final_stock_date = dt.datetime.now().date()

model_config = {
    "TEST_SIZE": 0.2,
    "TARGET_NAME": "Close",
    "VALIDATION_METRIC": "MAPE",
    "OPTIMIZATION_METRIC": "MSE",
    "FORECAST_HORIZON": 10
}

# Define a ação para procurar
STOCK_NAME = 'BOVA11.SA'