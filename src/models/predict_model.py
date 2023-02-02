import sys
sys.path.insert(0,'.')

from src.utils import *

# make the dataset
PERIOD = '800d'
INTERVAL = '1d'
STOCK_NAME = 'BOVA11.SA'  #str(input("Which stock do you want to track? "))

logger.info("Starting the inference pipeline..")

# load the raw dataset
stock_df = pd.read_csv("./data/raw/raw_stock_prices.csv", parse_dates=True)
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# perform featurization
features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]
stock_df_feat = build_features(stock_df, features_list)

def main():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.

    Parameters:
        None
    Returns:
        None
    """
    
    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)
    
    # Make predictions using the future dataframe and specified forecast horizon
    predictions_df = make_predict(
        forecast_horizon=model_config["FORECAST_HORIZON"]-4,
        future_df=future_df
    )

    # Print the predictions
    print(predictions_df)

    # display the predictions on web
    st.write(f"Here are the forecast for {STOCK_NAME}")
    st.write(predictions_df)

    # make the figure using plotly
    fig = px.line(
        predictions_df,
        x="Date",
        y="Forecast",
        title=f"Default XGBoost {model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}"
    )

    # plot it
    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Execute the whole pipeline
if __name__ == "__main__":

    main()

    logger.info("Inference Pipeline was sucessful!")