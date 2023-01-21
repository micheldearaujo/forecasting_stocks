# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf  # Plot de Autocorrelação - Moving Averages
from statsmodels.graphics.tsaplots import plot_pacf  # Plot de Autocorrelação - Auto Regressive
from pmdarima.arima.utils import ndiffs 
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use("fivethirtyeight")
# Define dates to start and end
initial_stock_date = dt.datetime.now().date() - dt.timedelta(days=3*365)
final_stock_date = dt.datetime.now().date()

# Define a ação para procurar
STOCK_NAME = 'BOVA11.SA'