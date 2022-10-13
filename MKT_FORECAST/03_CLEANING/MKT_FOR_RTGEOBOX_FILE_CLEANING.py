# Databricks notebook source
#!pip install openpyxl 

# COMMAND ----------

import os

import pandas as pd

# COMMAND ----------

DATA_ROOT = "mnt/adls/40_project/U70/FOR_ARTEFACT/MKT_FORECAST"
FILE_PATH = "RAW/NA Industry RT GEOBOX All Categories by Month by Year with Sub Category 24 Sept 2022.xlsx"

# COMMAND ----------

def read_file(folder_path, file_path):
    df = pd.read_excel("/dbfs/" + os.path.join(folder_path, file_path), header=2)
    return df

# COMMAND ----------

df = read_file(DATA_ROOT, FILE_PATH)

# COMMAND ----------

_df = df.copy()

# COMMAND ----------

df

# COMMAND ----------

get_year = lambda x: x[:4] + '-' + x[5:]

years=['2010', '2010.1', '2011.2', '2022.11']

list(map(get_year, years))

# COMMAND ----------

cols = list(df.columns)
cols[:16]

# COMMAND ----------

df.iloc[0, :16] = cols[:16]

# COMMAND ----------

df.rename(columns=df.iloc[0], inplace = True)

# COMMAND ----------

df.drop(0, inplace=True)

# COMMAND ----------

df.drop(47029, inplace=True)

# COMMAND ----------

df = df.loc[:, df.columns.notna()]

# COMMAND ----------

df_melted = df.melt(id_vars=cols[:17], var_name='DATE', value_name='SALES')

# COMMAND ----------

df_melted.to_parquet("/dbfs/" + os.path.join(DATA_ROOT, 'INTERIM/NA_INDUSTRY_RT_GEOBOX_SEP2022.parquet'))
