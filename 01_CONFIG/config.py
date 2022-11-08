# Databricks notebook source
# MAGIC %md
# MAGIC ## General Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

!pip install sklearn
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
import datetime as dt
import time

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
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

# Classical Machine Learning Libraries
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

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

from hyperopt import fmin, tpe, Trials, hp, SparkTrials, space_eval, STATUS_OK, rand, Trials
import mlflow
import mlflow.spark

!pip install lightgbm
import lightgbm as lgb
from lightgbm import LGBMRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Constants

# COMMAND ----------

model_config = {
    'NUMBER_OF_STD_TO_KEEP_OUTLIERS': 2,
    'TARGET_VARIABLE': 'LOG_TOTAL_PROCESSING_TIME',
    'TEST_SIZE_TEST': 0.2,
    'TEST_SIZE_VAL': 0.5,
}

etr_model_config = {
    'CRITERION': 'mse',
    'MAX_DEPTH': 60,
    'MIN_SAMPLES_LEAF': 2,
    'MIN_SAMPLES_SPLIT': 3,
    'N_ESTIMATORS': 400,    
}

lgbm_model_config = {
    'BOOSTING_TYPE': 'goss',
    'LEARNING_RATE': 0.15,
    'MAX_DEPTH': 40,
    'MIN_DATA': 100,
    'N_ESTIMATORS': 200,
    'NUM_LEAVES': 250,
    'REG_LAMBDA': 1,
    'SCALE_POS_WEIGHT': 1,
    'SEED': 42,
    'SUBSAMPLE': 0.9,
    'COLSAMPLE_BYTREE': 1.0,
    'VERBOSE': 20,
    'NUM_BOOST_ROUND': 300
}

lgbm_fixed_model_config = {
    'LEARNING_RATE': 0.1,
    'MIN_DATA': 100,
    'SEED': 42,
    'SUBSAMPLE': 0.9,
    'VERBOSE': 20,
    'NUM_ITERATIONS': 200
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

random_forest_model_config = {
    'bootstrap': False,
    'max_depth': 90,
    'max_features': 'auto',
    'min_samples_leaf': 1,
    'min_samples_split': 10,
    'n_estimators': 1000
}

xgboost_fixed_model_config = {
    'SEED': 42,
    'SUBSAMPLE': 0.95
}

random_forest_fixed_model_config = {
    'RANDOM_STATE': 42,
}

# COMMAND ----------

etr_hyperparameter_config = {
    'n_estimators':hp.choice('n_estimators',[80, 140, 150, 180, 200, 250, 300]),
    'max_depth':hp.choice('max_depth',[2, 10, 15, 30, 50, 100]),          
    'min_samples_split':hp.choice('min_samples_split',[3, 7, 9, 12, 15]),  
    'min_samples_leaf':hp.choice('min_samples_leaf',[2, 4, 5, 8, 12]),
    'criterion':hp.choice('criterion', ['mse'])
}

lightgbm_hyperparameter_config = {
    'num_leaves':hp.choice('num_leaves',[20, 40, 100, 240, 245, 250, 255]),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'n_estimators':hp.choice('n_estimators',[280, 290, 300, 310, 320]),
    'boosting_type':hp.choice('boosting_type',['dart', 'goss']),
    'max_depth':hp.choice('max_depth',[10, 20, 38, 40, 42]),
    'reg_lambda':hp.choice('reg_lambda',[0, 1, 2, 3, 10, 15]),
    'scale_pos_weight':hp.choice('scale_pos_weight',[1, 2, 3, 4, 10, 15]),
}

xgboost_hyperparameter_config = {
    'max_depth': hp.choice('max_depth', [9, 11, 30]),
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1, 0.5, 1.0]),
    'gamma': hp.choice('gamma', [0.01, 0.1, 1.0]),
    'reg_lambda': hp.choice('reg_lambda', [1, 10, 30, 100]),
    'n_estimators': hp.choice('n_estimators', [40, 200, 300]),
    'scale_pos_weight': hp.choice('scale_pos_weight', [2, 5, 8, 10]),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
}

randomforest_hyperparameter_config = {
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'max_depth': hp.choice('max_depth', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]),
    'max_features': hp.choice('max_features', ['auto', 'sqrt']),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 7, 10]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10, 20]),
    'n_estimators': hp.choice('n_estimators', [200, 400, 800, 1000, 1200, 1600, 2000])
}
