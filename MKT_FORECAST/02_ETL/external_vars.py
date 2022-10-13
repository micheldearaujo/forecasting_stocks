# Databricks notebook source
!pip install eiapy

# COMMAND ----------

import sys, os
os.environ['EIA_KEY'] = '81df175923880a66568e31bb53783b3d'
#https://www.eia.gov/opendata/qb.php

# COMMAND ----------

#import databricks.koalas as ks
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from eiapy import Series
from os import listdir

# COMMAND ----------

DATA_ROOT = "mnt/adls/40_project/SND/PROJECTS/MARKET_OE_RT/"
OUTPUT_FILE_PATH = "DATA_IN/external_vars_eia.csv"

# COMMAND ----------

start_date = '2017-01'
end_date = pd.to_datetime("now").strftime('%Y-%m')
nb_months = int(((pd.to_datetime(end_date) - pd.to_datetime(start_date))/np.timedelta64(1, 'M')))

# COMMAND ----------

date_start = '201001'
date_end ='202208'

def extract_data_eia_monthly(date_start, date_end, code, column_name):
  sam  = Series(code)
  d = sam.get_data(start=date_start, end=date_end)
  df = pd.DataFrame(d['series'][0]['data'], columns=['date', column_name])
  df['date'] = pd.to_datetime(df['date'].astype('str').apply(lambda x: '-'.join([x[:4], x[4:6]])))
  return df

#can find the codes here: https://www.eia.gov/opendata/qb.php

code_dict = {'STEO.DSRTUUS.M': 'retail_diesel_price_cents',
             'STEO.MGEIAUS.M': 'retail_gas_price_cents',
             'STEO.TETCCO2.M': 'co2_mill_tons',
             'STEO.XRUNR.M':   'unemplyment_percent',
             'STEO.HSTCXUS.M': 'housing_starts_mill',
             'STEO.GDPQXUS.M': 'real_gdp_bill_chained',
             'STEO.ACTKFUS.M': 'airline_ticket_price_index',
             'STEO.RSPRPUS.M': 'steel_production_mill_short_tons',
             'STEO.MVVMPUS.M': 'vehicle_miles_traveled_mill_miles_day',
             'STEO.CICPIUS.M': 'consumer_price_index',
              'PET.WGFUPUS2.W': 'US_WEEKLY' }

# df_eia = pd.DataFrame()

for i, (key, item) in enumerate(code_dict.items()):
  if i==0:
    df_eia = extract_data_eia_monthly(date_start, date_end, key, item)
  else:
    df_eia = pd.merge(df_eia, extract_data_eia_monthly(date_start, date_end, key, item),left_on='date',
    right_on='date')
  
df_eia.head(3)

# COMMAND ----------

df_eia_monthly = df_eia.groupby('date').mean()

# COMMAND ----------

df_eia_monthly.to_csv("/dbfs/" + os.path.join(DATA_ROOT, OUTPUT_FILE_PATH))

# COMMAND ----------


