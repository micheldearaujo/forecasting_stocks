# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *

logger = logging.getLogger("feature-engineering")
logger.setLevel(logging.DEBUG)


def build_features(raw_df: pd.DataFrame, features_list: list, save: bool=True) -> pd.DataFrame:
    """
    This function creates the features for the dataset to be consumed by the
    model
    
    :param raw_df: Raw Pandas DataFrame to create the features of
    :param features_list: The list of features to create

    :return: Pandas DataFrame with the new features
    """

    logger.debug("Building features...")
    final_df_featurized = pd.DataFrame()

    for stock_name in raw_df["Stock"].unique():
        logger.debug("Building features for stock %s..."%stock_name)
        stock_df_featurized = raw_df[raw_df['Stock'] == stock_name].copy()
        
        stock_df_featurized['day_of_month'] = stock_df_featurized["Date"].apply(lambda x: float(x.day))
        stock_df_featurized['month'] = stock_df_featurized['Date'].apply(lambda x: float(x.month))
        stock_df_featurized['quarter'] = stock_df_featurized['Date'].apply(lambda x: float(x.quarter))
        stock_df_featurized['week'] = stock_df_featurized['Date'].apply(lambda x: float(x.week))
        stock_df_featurized['Close'] = stock_df_featurized['Close'].apply(lambda x: round(x, 2))
        moving_averages_features = [feature for feature in features_list if "MA" in feature]
        for feature in moving_averages_features:
            ma_value = int(feature.split("_")[-1])
            stock_df_featurized[f'CLOSE_MA_{ma_value}'] = stock_df_featurized['Close'].rolling(ma_value, closed='left').mean()

        lag_features = [feature for feature in features_list if "lag" in feature]
        for feature in lag_features:
            lag_value = int(feature.split("_")[-1])
            stock_df_featurized[f'Close_lag_{lag_value}'] = stock_df_featurized['Close'].shift(lag_value)

        # Drop nan values because of the shift
        stock_df_featurized = stock_df_featurized.dropna()

        # Concatenate the new features to the final dataframe
        final_df_featurized = pd.concat([final_df_featurized, stock_df_featurized], axis=0)
        
    
    if save:
        
        final_df_featurized.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), index=False)

    logger.debug("Features built successfully!")
    print(final_df_featurized.tail())
    print(f"Dataset shape: {final_df_featurized.shape}.")
    print(f"Amount of ticker symbols: {final_df_featurized['Stock'].nunique()}.")

    return final_df_featurized



if __name__ == '__main__':

    logger.debug("Loading the raw dataset to featurize it...")
    stock_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), parse_dates=['Date'])

    logger.info("Featurizing the dataset...")
    stock_df_feat = build_features(stock_df, features_list)
    

    logger.info("Finished featurizing the dataset!")