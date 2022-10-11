# Databricks notebook source
# MAGIC %md
# MAGIC ## General Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

!pip install statsmodels
!pip install pmdarima
!pip install xgboost

# COMMAND ----------

import pyspark
from pyspark import SparkFiles
import pyspark.sql.functions as F

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller  # Teste de estacionaridade
from statsmodels.graphics.tsaplots import plot_acf  # Plot de Autocorrelação - Moving Averages
from statsmodels.graphics.tsaplots import plot_pacf  # Plot de Autocorrelação - Auto Regressive
from pmdarima.arima.utils import ndiffs  # Testes para saber o número de diferenciações
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from pmdarima.arima.utils import ndiffs 
from statsmodels.tsa.seasonal import seasonal_decompose

# Classical Machine Learning Libraries
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score

# Configuration
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use("fivethirtyeight")



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Constants

# COMMAND ----------

DATASET_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
URL_DATASET_NAME = 'daily-min-temperatures.csv'
DATA_PATH = '/tmp/data'
RAW_DATASET_NAME = 'raw_temperatures.csv'
INTERIM_DATASET_NAME = 'interim_temperatures.csv'

# COMMAND ----------


