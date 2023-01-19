# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import date
import pandas_datareader as web
import yfinance as yfin
import datetime as dt

plt.style.use("fivethirtyeight")
# Define dates to start and end
initial_stock_date = dt.datetime.now().date() - dt.timedelta(days=3*365)
final_stock_date = dt.datetime.now().date()

# Define a ação para procurar
STOCK_NAME = 'BOVA11.SA'
# Para usar datas:
#start=initial_stock_date, end=final_stock_date, 
stock_price_df = yfin.Ticker(STOCK_NAME).history(period= '720d', interval='1d')
stock_price_df['Stock'] = STOCK_NAME
stock_price_df = stock_price_df[['Close']]
stock_price_df = stock_price_df.reset_index()
print(stock_price_df.head())
stock_price_df.to_csv('./data/raw/raw_stock_prices.csv', index=False)