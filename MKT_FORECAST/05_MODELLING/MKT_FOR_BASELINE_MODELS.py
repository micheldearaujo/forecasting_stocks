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

df['MONTH_STR'] = df['DATE'].apply(lambda x: x[:3])
df['MONTH'] = df['MONTH_STR'].map(month_map)
df['YEAR'] = df['DATE'].apply(lambda x: int(x[3:]))
df['DAY'] = 1
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])


# COMMAND ----------

df

# COMMAND ----------

df = df[df['DATE'] != pd.to_datetime('2022-08-01')]

# COMMAND ----------

#df = df[(df.YEAR != int(2022)) & (df.MONTH != 8)]

# COMMAND ----------

print( f'Available years: {(df.DATE.max() - df.DATE.min()).days / 365}')

# COMMAND ----------

ax = plt.gca()
for country in df['Country'].unique():
    df[df['Country'] == country].groupby('DATE').agg({'SALES': 'sum'}).plot(ax=ax)
plt.legend(df['Country'].unique())
plt.xlim('2020-01-01','2022-07-01')

# COMMAND ----------


