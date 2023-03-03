import sys
sys.path.insert(0,'.')

from src.utils import *


def front_end():
    """
    Main function that creates a future dataframe, makes predictions, and prints the predictions.

    Parameters:
        None
    Returns:
        None
    """
    st.write("""# Welcome to the Stock Forecaster!
    Here you can have a glance of how the future of your stocks can look like and *simulute* decisions based on that.
    Please keep in mind that this is just a educational tool and you should not perform financial operations based on that.
    """)

    st.sidebar.write("""#### Choose your filters""")

    STOCK_NAME = st.sidebar.selectbox(
        "Which stock do you want to track?",
        ("BOVA11.SA", "ITUB4.SA", "VALE3.SA", "NFLX")
    )

    hist_start_date = st.sidebar.date_input(
        "From when do you want to the the historical prices?",
        dt.datetime.today().date() - dt.timedelta(days=2*model_config["FORECAST_HORIZON"])
    )

    historical_color = st.sidebar.radio(
        "Select historical line color 👇",
        ["blue", "red", "black", "orange", "green"],
        key="his_color",
        label_visibility="visible",
        horizontal=True,
    )

    forecast_color = st.sidebar.radio(
        "Select Forecast line color 👇",
        ["blue", "red", "black", "orange", "green"],
        key="for_color",
        label_visibility="visible",
        horizontal=True,
    )
    # load the predictions dataset
    predictions_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'output_stock_prices.csv'), parse_dates=["Date"])
    # filter the predictions dataset to only the stock
    predictions_df = predictions_df[predictions_df["Stock"] == STOCK_NAME]
    # load the historical dataset
    historical_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])
    # filter the historical dataset to only the stock
    historical_df = historical_df[historical_df["Stock"] == STOCK_NAME]
    # filter the last 10 days of the historical dataset and concat
    historical_df = historical_df[historical_df["Date"] >= pd.to_datetime(hist_start_date)]

    full_df = pd.concat([historical_df, predictions_df], axis=0).reset_index().fillna(0)
    full_df["Class"] = full_df["Close"].apply(lambda x: "Historical" if x > 0 else "Forecast")
    full_df["Price"] = full_df["Close"] + full_df["Forecast"]
    full_df = full_df[['Date', 'Class', 'Price']]
    print(full_df)

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


    # display the predictions on web
    col1, col2, col3 = st.columns(3)
    col2.metric(label="Mininum price", value=f"R$ {round(predictions_df['Forecast'].min(), 2)}")
    col1.metric(label="Maximum price", value=f"R$ {round(predictions_df['Forecast'].max(), 2)}")
    col3.metric(label="Amplitude", value=f"R$ {round(predictions_df['Forecast'].max() - predictions_df['Forecast'].min(), 2)}",
                delta=f"{100*round((predictions_df['Forecast'].max() - predictions_df['Forecast'].min())/predictions_df['Forecast'].min(), 4)}%")



# Execute the whole pipeline
if __name__ == "__main__":

    front_end()