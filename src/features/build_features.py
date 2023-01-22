# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/michel/Projects/mlops_practice/src')
from utils import *

# Load the dataset
stock_df = pd.read_csv('./data/raw/raw_stock_prices.csv')
print(stock_df.info())
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
stock_df['Date'] = stock_df['Date'].apply(lambda x: x.date())
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
print(stock_df.info())