import sys
sys.path.insert(0,'.')

from src.utils import *



def main():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.

    Parameters:
        None
    Returns:
        None
    """

    #STOCK_NAME = 'BOVA11.SA'  #str(input("Which stock do you want to track? "))
    STOCK_NAME = st.selectbox(
        "Which stock do you want to track?",
        ("BOVA11.SA", "ITUB4.SA", "VALE3.SA", "NFLX")
    )
    st.write("You selected:", STOCK_NAME)

    # load the predictions dataset
    predictions_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), parse_dates=["Date"])
    # load the historical dataset
    historical_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    # filter the last 10 days of the historical dataset and concat
    full_df = pd.concat([historical_df.tail(model_config["FORECAST_HORIZON"]), predictions_df], axis=0).reset_index().fillna(0)
    full_df["Class"] = full_df["Close"].apply(lambda x: "Historical" if x > 0 else "Forecast")
    full_df["Price"] = full_df["Close"] + full_df["Forecast"]
    full_df = full_df[['Date', 'Class', 'Price']]
    print(full_df)

    # # display the predictions on web
    st.write(f"{model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}")
    st.write(full_df)

    # make the figure using plotly
    fig = px.line(
        full_df,
        x="Date",
        y="Price",
        color="Class",
        symbol="Class",
        title=f"{model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}"
    )

    # # plot it
    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Execute the whole pipeline
if __name__ == "__main__":

    main()