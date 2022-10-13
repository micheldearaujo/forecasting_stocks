# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration
# MAGIC 
# MAGIC **Objective**: The purpose of this notebook is to hold all constants definition, business rules and libraries importing

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as Types


import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import RobustScaler

plt.style.use('fivethirtyeight')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Constants

# COMMAND ----------

DATA_ROOT = "mnt/adls/40_project/SND/PROJECTS/MARKET_OE_RT/"
MKT_FILE_PATH = "INTERIM/NA_INDUSTRY_RT_GEOBOX_SEP2022.parquet"
EXT_VARS_FILE_PATH = "DATA_IN/external_vars_eia.csv"

PROCESSED_MKT_FILE_PATH = "INTERIM/SEMI-PROCESSED_EXTERNAL_SALES.parquet"
