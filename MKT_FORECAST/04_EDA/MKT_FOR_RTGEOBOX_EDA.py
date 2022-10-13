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

#display(df)

# COMMAND ----------

print( f'Available years: {(df.DATE.max() - df.DATE.min()).days / 365}')

# COMMAND ----------

df[df['DATE'] == '2022-08-01']

# COMMAND ----------

7*2*26

# COMMAND ----------

df.groupby('DATE').agg({'SALES': 'sum'}).plot()

# COMMAND ----------

ax = plt.gca()
for country in df['Country'].unique():
    df[df['Country'] == country].groupby('DATE').agg({'SALES': 'sum'}).plot(ax=ax)
plt.legend(df['Country'].unique())

# COMMAND ----------

sns.countplot('NA Industry Product Category', data=df, hue='Country')

# COMMAND ----------

sns.countplot('NA Industry Groupe Category', data=df, hue='Country')

# COMMAND ----------

CATEGORY = 'NA Industry Groupe Category'
ax = plt.gca()
for cat in df[CATEGORY].unique():
    df[(df['Country'] == 'UNITED STATES') & (df[CATEGORY] == cat)].groupby(['DATE']).agg({'SALES': 'sum'}).plot(ax=ax)
    
plt.legend(df[CATEGORY].unique())

# COMMAND ----------

CATEGORY = 'NA Industry Groupe Category'
ax = plt.gca()
for cat in df[CATEGORY].unique():
    df[(df['Country'] == 'CANADA') & (df[CATEGORY] == cat)].groupby(['DATE']).agg({'SALES': 'sum'}).plot(ax=ax)
    
plt.legend(df[CATEGORY].unique())

# COMMAND ----------

CATEGORY = 'NA Industry Product Category'
ax = plt.gca()
for cat in df[CATEGORY].unique():
    df[df[CATEGORY] == cat].groupby(['DATE']).agg({'SALES': 'sum'}).plot(ax=ax)
    
plt.legend(df[CATEGORY].unique())

# COMMAND ----------

data = df.groupby('DATE').agg({'SALES': 'sum'})

# COMMAND ----------

from statsmodels.tsa.seasonal import STL

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
res = STL(data).fit()
res.plot()
plt.show()

# COMMAND ----------

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize = (16,6))
plot_acf(data, lags=40, ax=ax1)
plot_pacf(data, lags=40, ax=ax2)

# COMMAND ----------

from statsmodels.tsa.api import ExponentialSmoothing

train = data[:-12]
test = data[-12:]

holt_winters_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
pred = holt_winters_model.forecast(12)

# COMMAND ----------

test.plot()
pred.plot()
plt.legend(['actual', 'forecast'])

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
train.plot(ax=ax)
test.plot(ax=ax)
pred.plot(ax=ax)
plt.legend(['train', 'test', 'forecast'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Modeling NA Industry Groupe Category

# COMMAND ----------

def model_industry_groupe_category(cat, horizon=12):
    cat_data = df[df['NA Industry Groupe Category'] == cat].groupby(['DATE']).agg({'SALES': 'sum'})
    
    train = cat_data[:-horizon]
    test = cat_data[-horizon:]

    holt_winters_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
    pred = holt_winters_model.forecast(horizon)
    
    ax = plt.gca()
    train.plot(ax=ax, figsize=(16, 6))
    test.plot(ax=ax, figsize=(16, 6))
    pred.plot(ax=ax, figsize=(16, 6))
    plt.title(f'Holt-Winters model for {cat} industry groupe category')
    plt.legend(['train', 'test', 'forecast'])

# COMMAND ----------

df['NA Industry Groupe Category'].unique()

# COMMAND ----------

model_industry_groupe_category('Z')

# COMMAND ----------

model_industry_groupe_category('V')

# COMMAND ----------

model_industry_groupe_category('H')

# COMMAND ----------

model_industry_groupe_category('ENTRY (S,T)')


# COMMAND ----------

model_industry_groupe_category('WINTER')

# COMMAND ----------

model_industry_groupe_category('RECREATIONAL')

# COMMAND ----------

model_industry_groupe_category('COMMERCIAL')


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Modeling NA Industry Groupe Category
