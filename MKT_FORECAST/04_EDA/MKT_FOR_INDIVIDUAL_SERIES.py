# Databricks notebook source
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (18,4)

# COMMAND ----------

DATA_ROOT = "mnt/adls/40_project/SND/PROJECTS/MARKET_OE_RT/"
FILE_PATH = "INTERIM/NA_INDUSTRY_RT_GEOBOX_SEP2022.parquet"


# COMMAND ----------

df = pd.read_parquet("/dbfs/" + os.path.join(DATA_ROOT, FILE_PATH))

# COMMAND ----------

month_map = {
    'JAN': 1,
    'FEB': 2,
    'MAR': 3,
    'APR': 4,
    'MAY': 5,
    'JUN': 6,
    'JUL': 7,
    'AUG': 8,
    'SEP': 9,
    'OCT': 10,
    'NOV': 11,
    'DEC': 12
}

# COMMAND ----------

# MAGIC %md Groupe Category US

# COMMAND ----------

df[(df['Country'] == 'UNITED STATES') & (df[CATEGORY] == cat)].groupby(['DATE']).agg({'SALES': 'sum'})

# COMMAND ----------

# MAGIC %md Groupe Category Canada

# COMMAND ----------

df[(df['Country'] == 'CANADA') & (df[CATEGORY] == cat)].groupby(['DATE']).agg({'SALES': 'sum'})

# COMMAND ----------

# MAGIC %md Industry category

# COMMAND ----------

df[df[CATEGORY] == cat].groupby(['DATE']).agg({'SALES': 'sum'})

# COMMAND ----------


