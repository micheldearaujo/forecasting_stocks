# -*- coding: utf-8 -*-

sys.path.insert(0,'.')

from src.utils import *



def make_dataset(stock_name: str, period: str, interval: str):
    """
    Creates a dataset of the closing prices of a given stock.
    
    Parameters:
        stock_name (str): The name of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing prices of the given stock.
    """

    stock_price_df = yfin.Ticker(stock_name).history(period=period, interval=interval)
    stock_price_df['Stock'] = stock_name
    stock_price_df = stock_price_df[['Close']]
    stock_price_df = stock_price_df.reset_index()
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df['Date'] = stock_price_df['Date'].apply(lambda x: x.date())
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df.to_csv('./data/raw/raw_stock_prices.csv', index=False)

    return stock_price_df