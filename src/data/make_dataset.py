# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

import logging.config
import yaml

from src.utils import *

with open("src/configuration/logging_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)


def fetch_stock_price_data(stock_name: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download data of the closing prices of a given stock and return as Pandas DataFrame.
    
    Parameters:
        stock_name (str): The ticker symbol of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing price of a single ticker symbol.
    """
    logger.debug(f"Downloading data for Ticker: {stock_name}...")

    stock_price_df = yfin.Ticker(stock_name).history(period=period, interval=interval)
    
    stock_price_df['Stock'] = stock_name
    stock_price_df = stock_price_df[['Stock', 'Close']]
    stock_price_df = stock_price_df.reset_index()

    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
    stock_price_df['Date'] = stock_price_df['Date'].apply(lambda x: x.date())
    stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])

    return stock_price_df


def make_dataset(stock_name: str, period: str, interval: str, save_to_table: bool = True) -> pd.DataFrame:
    """
    Creates a dataset of the closing prices of a given stock.
    
    Parameters:
        stock_name (str): The name of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing prices all stocks.
    """
    raw_df = pd.DataFrame()
    
    for stock_name in stocks_list:

        stock_price_df = fetch_stock_price_data(stock_name=stock_name, period=period, interval=interval)

        raw_df = pd.concat([raw_df, stock_price_df], axis=0)

    # Save the dataset
    if save_to_table:
        raw_df.to_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), index=False)

    return raw_df


if __name__ == '__main__':


    logger.info("Downloading the raw dataset...")

    stock_df = make_dataset(stocks_list, PERIOD, INTERVAL)

    logger.info("Finished downloading the raw dataset!")