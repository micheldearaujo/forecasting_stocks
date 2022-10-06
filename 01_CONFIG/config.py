# Databricks notebook source
# MAGIC %md
# MAGIC ## General Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

import pyspark
from pyspark import SparkFiles

import os

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


