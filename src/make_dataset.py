# -*- coding: utf-8 -*-


from utils import *


def make_dataset(stock_name: str, period: str, interval: str):

    stock_price_df = yfin.Ticker(stock_name).history(period=period, interval=interval)
    stock_price_df['Stock'] = stock_name
    stock_price_df = stock_price_df[['Close']]
    stock_price_df = stock_price_df.reset_index()
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df['Date'] = stock_price_df['Date'].apply(lambda x: x.date())
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df.to_csv('./data/raw/raw_stock_prices.csv', index=False)