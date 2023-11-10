# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *

logger = logging.getLogger("Make_dataset")
logger.setLevel(logging.INFO)

def make_dataset(stock_name: str, period: str, interval: str) -> pd.DataFrame:
    """
    Creates a dataset of the closing prices of a given stock.
    
    Parameters:
        stock_name (str): The name of the stock to retrieve data for.
        period (str): The length of time to retrieve data for, e.g. '1d', '1mo', '3mo', '6mo', '1y', '5y', 'max'.
        interval (str): The frequency of the data, e.g. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'.
    
    Returns:
        pandas.DataFrame: The dataframe containing the closing prices of the given stock.
    """
    empty_df = pd.DataFrame()
    
    for stock_name in stocks_list:

        stock_price_df = yfin.Ticker(stock_name).history(period=period, interval=interval)
        stock_price_df['Stock'] = stock_name
        stock_price_df = stock_price_df[['Stock', 'Close']]
        stock_price_df = stock_price_df.reset_index()
        stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])
        stock_price_df['Date'] = stock_price_df['Date'].apply(lambda x: x.date())
        stock_price_df['Date'] = pd.to_datetime(stock_price_df['Date'])

        empty_df = pd.concat([empty_df, stock_price_df], axis=0)

    empty_df.to_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), index=False)

    return stock_price_df


if __name__ == '__main__':

    logger.info("Downloading the raw dataset...")

    stock_df = make_dataset(stocks_list, PERIOD, INTERVAL)

    logger.info("Finished downloading the raw dataset!")