# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/michel/Projects/mlops_practice/src')
from utils import *


# Para usar datas:
#start=initial_stock_date, end=final_stock_date, 
stock_price_df = yfin.Ticker(STOCK_NAME).history(period= '720d', interval='1d')
stock_price_df['Stock'] = STOCK_NAME
stock_price_df = stock_price_df[['Close']]
stock_price_df = stock_price_df.reset_index()
print(stock_price_df.tail())
stock_price_df.to_csv('./data/raw/raw_stock_prices.csv', index=False)
stock_price_df.set_index('Date').plot()
plt.show()  