# Databricks notebook source
# MAGIC %md
# MAGIC # External variables EDA
# MAGIC 
# MAGIC **Objective**: The purpose of this notebook is 
# MAGIC - Load the external variables related to the Sales volume and check if they are statistically usefull for using them to predict the target;
# MAGIC - Some tests to be made:
# MAGIC   - Check Stationarity;
# MAGIC   - Check differences;
# MAGIC   - Check Granger Causality;
# MAGIC 
# MAGIC **Takeaways**: Some of the conclusions about this notebook:
# MAGIC 
# MAGIC ##### Results of the Stationarity and Granger tests:
# MAGIC 
# MAGIC | Variable | Stationary of diff (d) | Granger lag |
# MAGIC |:--------:|:------------------:|:-----------:|
# MAGIC | SALES (the target) | 1 |  |
# MAGIC | retail_diesel_price_cents | 1 |  |
# MAGIC | retail_gas_price_cents |1 |  |
# MAGIC | co2_mill_tons | 2 |  |
# MAGIC | unemplyment_percent | 1 |  |
# MAGIC | housing_starts_mill | 1 |  |
# MAGIC | real_gdp_bill_chained | 1 |  |
# MAGIC | airline_ticket_price_index | 2 |  |
# MAGIC | steel_production_mill_short_tons | 0 |  |
# MAGIC | vehicle_miles_traveled_mill_miles_day | 1 |  |
# MAGIC | consumer_price_index | 2 |  |
# MAGIC | US_WEEKLY | 0 |  |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Reading

# COMMAND ----------

mkt = pd.read_parquet("/dbfs/" + os.path.join(DATA_ROOT, MKT_FILE_PATH))
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
mkt['MONTH_STR'] = mkt['DATE'].apply(lambda x: x[:3])
mkt['MONTH'] = mkt['MONTH_STR'].map(month_map)
mkt['YEAR'] = mkt['DATE'].apply(lambda x: int(x[3:]))
mkt['DAY'] = 1
mkt['DATE'] = pd.to_datetime(mkt[['YEAR', 'MONTH', 'DAY']])
mkt.rename(columns={'DATE': 'date'}, inplace=True)
mkt.head()

# COMMAND ----------

mkt.info()

# COMMAND ----------

ext = pd.read_csv("/dbfs/" + os.path.join(DATA_ROOT, EXT_VARS_FILE_PATH), parse_dates=True)
ext['date'] = pd.to_datetime(ext['date'])
ext.set_index('date', inplace=True)
ext.head()

# COMMAND ----------

ext.info()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3.0 Filtering US MARKET Sellin

# COMMAND ----------

us_total_mkt = mkt[mkt['Country'] == 'UNITED STATES'].groupby('date').agg({'SALES': 'sum'})
us_total_mkt.drop('2022-08-01', inplace=True)
us_total_mkt.head()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
us_total_mkt.plot(ax=ax)
plt.title("US Sellin")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 4.0 Stationarity Test for the *US Market Sell-in*

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Testing the original Series

# COMMAND ----------

adf_test = adfuller(us_total_mkt)
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The orinal series is not Stationary!!!!!

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Testing the difference

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
us_total_mkt.diff().plot(ax=ax)
plt.title("US Sellin First Difference")
plt.show()

# COMMAND ----------

# First diff stationarity test
adf_test = adfuller(us_total_mkt['SALES'].diff()[1:])
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The first difference is Stationary. So the `d` factor is 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 External Vars

# COMMAND ----------

# Join the external variables dataframe with the us_total_mkt dataframe (the target feature)
ext = ext.merge(us_total_mkt, left_index=True, right_index=True)

# COMMAND ----------

ext.head()

# COMMAND ----------

corr = ext.corr()

# COMMAND ----------

# plotting the heatmap
plt.figure(figsize=(10,8))
ax=plt.gca()
hm = sns.heatmap(data=corr, annot=True, ax=ax)
hm.set_xticklabels(hm.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

# COMMAND ----------

list(ext.columns)

# COMMAND ----------

scaler = RobustScaler()
scaled_ext = ext.copy()
scaled_ext[['retail_diesel_price_cents', 'retail_gas_price_cents', 'co2_mill_tons',
       'unemplyment_percent', 'housing_starts_mill', 'real_gdp_bill_chained',
       'airline_ticket_price_index', 'steel_production_mill_short_tons',
       'vehicle_miles_traveled_mill_miles_day', 'consumer_price_index',
       'US_WEEKLY', 'SALES']] = scaler.fit_transform(scaled_ext[['retail_diesel_price_cents', 'retail_gas_price_cents', 'co2_mill_tons',
       'unemplyment_percent', 'housing_starts_mill', 'real_gdp_bill_chained',
       'airline_ticket_price_index', 'steel_production_mill_short_tons',
       'vehicle_miles_traveled_mill_miles_day', 'consumer_price_index',
       'US_WEEKLY', 'SALES']])

# COMMAND ----------

# MAGIC %md 
# MAGIC #### 5.1 Vehicle Miles

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['vehicle_miles_traveled_mill_miles_day'].plot(ax=ax)
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.vehicle_miles_traveled_mill_miles_day[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['scaled_vehicle_miles_traveled_mill_miles_day', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.1.1 Stationarity Test

# COMMAND ----------

# Stationarity Test
x_label = 'vehicle_miles_traveled_mill_miles_day'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original Series is NOT stationary!!!

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext[x_label].diff().plot(ax=ax)
ax.set_title("Vehicles Miles Diff(1) series stationarity test")
plt.show()

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The FIRST difference is stationary!!!

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.1.2 Granger Causality Test

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]].diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: Being both series Diff(1) stationary, the Granger test points bellow 0.05 for all Lags. 
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'scaled_vehicle_miles_traveled_mill_miles_day_first_diff'])
ax.set_title("Vehicle miles diff(1) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 Retail Gas Price Cents

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['retail_gas_price_cents'].plot(ax=ax)
ax.set_title("Retail Gas Price Cents")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.retail_gas_price_cents[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['retail_gas_price_cents', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.2.1 Stationarity Test

# COMMAND ----------

# Stationarity Test
x_label = 'retail_gas_price_cents'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original Series is NOT Stationary!!

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext[x_label].diff().plot(ax=ax)
ax.set_title("Retail Gas Price Diff(1) series stationarity test")
plt.show()

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The diff(1) is Stationary!

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]].diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: Being both series Diff(1) stationary, the Granger test points bellow 0.05 for all Lags 1, 2, 3, and 12. 
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'retail_gas_price_cents_first_diff'])
ax.set_title("Retail gas price diff(1) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.3 US_WEEKLY (Gasoline supplied volume)

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['US_WEEKLY'].plot(ax=ax)
ax.set_title("US_WEEKLY")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.US_WEEKLY[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['US_WEEKLY', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.3.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'US_WEEKLY'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original series IS stationary!

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]][1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: The X series (US_WEEKLY) is Diff(0) stationary as the Y is Diff(1) stationary, the Granger test points bellow 0.05 for all Lags except for lag 1 
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'US_WEEKLY_first_diff'])
ax.set_title("US WEEKLY diff(0) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.4 consumer_price_index

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['consumer_price_index'].plot(ax=ax)
ax.set_title("consumer_price_index")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.consumer_price_index[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['consumer_price_index', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.4.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'consumer_price_index'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original series is NOT stationary!!

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext[x_label].diff().plot(ax=ax)
ax.set_title("consumer_price_index Diff(1) series stationarity test")
plt.show()

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The Diff(1) is NOT stationary!

# COMMAND ----------

# Stationarity Test of 2ª difference
adf_test = adfuller(ext[x_label].diff().diff()[2:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The Diff(2) IS stationary!

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext[x_label].diff().diff().plot(ax=ax)
ax.set_title("consumer_price_index Diff(2) series stationarity test")
plt.show()

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[2:-1]
df_test['x'] = scaled_ext[[x_label]].diff().diff()[2:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: The X series (US_WEEKLY) is Diff(2) stationary as the Y is Diff(1) stationary, the Granger test points bellow 0.05 for all Lags except 7.
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'consumer_price_index_first_diff'])
ax.set_title("consumer_price_index diff(2) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.5 unemplyment_percent

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['unemplyment_percent'].plot(ax=ax)
ax.set_title("unemplyment_percent")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.unemplyment_percent[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['unemplyment_percent', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.5.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'unemplyment_percent'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original series is NOT stationary!

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext[x_label].diff().plot(ax=ax)
ax.set_title("unemplyment_percent Diff(1) series stationarity test")
plt.show()

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The Diff(1) IS stationary!

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]].diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'unemplyment_percent_first_diff'])
ax.set_title("unemplyment_percent diff(1) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: Both series are  Diff(1) stationary, the Granger test points bellow 0.05 for Lags, 3, 4, 5, 6, 7, 11 and 12.
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.6 housing_starts_mill

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['housing_starts_mill'].plot(ax=ax)
ax.set_title("housing_starts_mill")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.housing_starts_mill[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['housing_starts_mill', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.6.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'housing_starts_mill'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original series is NOT stationary

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The Diff(1) IS stationary

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'housing_starts_mill_first_diff'])
ax.set_title("housing_starts_mill diff(1) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Test Granger for Diff(1)

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]].diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: Both series are Diff(1) stationary, the Granger test points bellow 0.05 for Lags 4, 5, 11 and 12.
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.7 airline_ticket_price_index

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['airline_ticket_price_index'].plot(ax=ax)
ax.set_title("airline_ticket_price_index")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.airline_ticket_price_index[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['airline_ticket_price_index', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.7.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'airline_ticket_price_index'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The original series is not Stationary

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# Stationarity Test of 2ª difference
adf_test = adfuller(ext[x_label].diff().diff()[2:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - The Diff(2) IS stationary

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'airline_ticket_price_index_first_diff'])
ax.set_title("airline_ticket_price_index diff(2) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Test Granger for Diff(2)

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[2:-1]
df_test['x'] = scaled_ext[[x_label]].diff().diff()[2:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: The X series (airline_ticket_price_index) is Diff(2) stationary as the Y is Diff(1) stationary, the Granger test points bellow 0.05 for all Lags.
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.8 steel_production_mill_short_tons

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['steel_production_mill_short_tons'].plot(ax=ax)
ax.set_title("steel_production_mill_short_tons")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.steel_production_mill_short_tons[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['steel_production_mill_short_tons', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.8.1 Stationarity tst

# COMMAND ----------

# Stationarity Test
x_label = 'steel_production_mill_short_tons'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The original series is Stationary

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Granger test on original series

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]]#.diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Granger test on Diff(1)

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]].diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.9 real_gdp_bill_chained

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['real_gdp_bill_chained'].plot(ax=ax)
ax.set_title("real_gdp_bill_chained")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.real_gdp_bill_chained[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['real_gdp_bill_chained', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.9.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'real_gdp_bill_chained'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The original series is NOT stationary

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The diff(1) IS stationary

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[1:-1]
df_test['x'] = scaled_ext[[x_label]].diff()[1:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.10 co2_mill_tons

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['co2_mill_tons'].plot(ax=ax)
ax.set_title("co2_mill_tons")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.co2_mill_tons[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['co2_mill_tons', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.10.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'co2_mill_tons'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ###### The original series is NOT starionary

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff().diff()[2:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The diff(2) is stationary

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[2:-1]
df_test['x'] = scaled_ext[[x_label]].diff().diff()[2:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'co2_mill_tons_first_diff'])
ax.set_title("co2_mill_tons diff(2) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.11 retail_diesel_price_cents

# COMMAND ----------

plt.figure(figsize=(10,8))
ax=plt.gca()
ext['retail_diesel_price_cents'].plot(ax=ax)
ax.set_title("retail_diesel_price_cents")
plt.show()

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
scaled_ext.retail_diesel_price_cents[:-1].plot(ax=ax)
scaled_ext.SALES[:-1].plot(ax=ax)
plt.legend(['retail_diesel_price_cents', 'scaled_us_sellin'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5.11.1 Stationarity test

# COMMAND ----------

# Stationarity Test
x_label = 'retail_diesel_price_cents'
y_label = 'SALES'

adf_test = adfuller(ext[x_label],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# Stationarity Test of 1ª difference
adf_test = adfuller(ext[x_label].diff()[1:],autolag='AIC')
dfoutput=pd.Series(adf_test[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
dfoutput

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The diff(2) is stationary

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Granger test

# COMMAND ----------

#Granger causality test
test='ssr_chi2test'
maxlag = 12
df_test = pd.DataFrame()
df_test['y'] = scaled_ext[[y_label]].diff()[2:-1]
df_test['x'] = scaled_ext[[x_label]].diff().diff()[2:-1]
test_result = grangercausalitytests(df_test[['y', 'x']], maxlag=maxlag, verbose=False)

p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
p_values_by_lag = { (idx + 1): el for idx, el in enumerate(p_values)}
p_values_by_lag

# COMMAND ----------

plt.figure(figsize=(16,6))
ax = plt.gca()
df_test.y.plot(ax=ax)
df_test.x.plot(ax=ax)
plt.legend(['scaled_us_sellin_first_diff', 'retail_diesel_price_cents_first_diff'])
ax.set_title("retail_diesel_price_cents diff(2) and sell-in diff(1)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### - Conclusion: Being both series Diff(1) stationary, the Granger test points bellow 0.05 for all Lags. 
# MAGIC 
# MAGIC Question: Which lag should we use?
# MAGIC 
# MAGIC Question: Do we need to specify a lag? Why not just use the column as it is?

# COMMAND ----------

ext

# COMMAND ----------

ext.to_parquet("/dbfs/" + os.path.join(DATA_ROOT, PROCESSED_MKT_FILE_PATH))
ext[["SALES"]].to_parquet("/dbfs/" + os.path.join(DATA_ROOT, PROCESSED_MKT_FILE_PATH))

# COMMAND ----------

ext_sdf = spark.createDataFrame(ext.reset_index())

# COMMAND ----------

ext_sdf.write.mode("overwrite").format("delta").saveAsTable("mkt_forecast_multivariate")
ext_sdf.select("date", "SALES").write.mode("overwrite").format("delta").saveAsTable("mkt_forecast_univariate")

# COMMAND ----------

display(ext_sdf)

# COMMAND ----------


