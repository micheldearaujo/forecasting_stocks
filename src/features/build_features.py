# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *


def build_features(raw_df: pd.DataFrame, features_list: list, save: bool=True) -> pd.DataFrame:
    """
    This function creates the features for the dataset to be consumed by the
    model
    
    :param raw_df: Raw Pandas DataFrame to create the features of
    :param features_list: The list of features to create

    :return: Pandas DataFrame with the new features
    """

    logger.debug("Started building features...")
    #stock_df_featurized = raw_df.copy()
    final_df_featurized = pd.DataFrame()

    print(raw_df.columns)

    for stock_name in raw_df["Stock"].unique():
        logger.debug("Building features for stock %s..."%stock_name)
        stock_df_featurized = raw_df[raw_df['Stock'] == stock_name].copy()
        
        for feature in features_list:
            
            # create "Time" features
            if feature == "day_of_month":
                stock_df_featurized['day_of_month'] = stock_df_featurized["Date"].apply(lambda x: float(x.day))
            elif feature == "month":
                stock_df_featurized['month'] = stock_df_featurized['Date'].apply(lambda x: float(x.month))
            elif feature == "quarter":
                stock_df_featurized['quarter'] = stock_df_featurized['Date'].apply(lambda x: float(x.quarter))

        # Create "Lag" features
        # The lag 1 feature will become the main regressor, and the regular "Close" will become the target.
        # As we saw that the lag 1 holds the most aucorrelation, it is reasonable to use it as the main regressor.
            elif feature == "Close_lag_1":
                stock_df_featurized['Close_lag_1'] = stock_df_featurized['Close'].shift()

            # Drop nan values because of the shift
            stock_df_featurized = stock_df_featurized.dropna()
            print(stock_df_featurized.tail(5))

        # Concatenate the new features to the final dataframe
        final_df_featurized = pd.concat([final_df_featurized, stock_df_featurized], axis=0)

    try:
        logger.debug("Rounding the features to 2 decimal places...")
        # handle exception when building the future dataset
        final_df_featurized['Close'] = final_df_featurized['Close'].apply(lambda x: round(x, 2))
        final_df_featurized['Close_lag_1'] = final_df_featurized['Close_lag_1'].apply(lambda x: round(x, 2))
        
    except KeyError:
        pass
    
    if save:
        print(final_df_featurized.tail(15))
        
        final_df_featurized.to_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), index=False)

    logger.debug("Features built successfully!")

    return final_df_featurized



if __name__ == '__main__':

    logger.debug("Loading the raw dataset to featurize it...")
    stock_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'raw_stock_prices.csv'), parse_dates=['Date'])

    logger.info("Featurizing the dataset...")
    stock_df_feat = build_features(stock_df, features_list)

    logger.info("Finished featurizing the dataset!")