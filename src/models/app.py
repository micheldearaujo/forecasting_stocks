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
        ("BOVA11.SA")#, "ITUB4.SA", "VALE3.SA", "NFLX")
    )
    st.write("You selected:", STOCK_NAME)

    # load the predictions dataset
    predictions_df = pd.read_csv("./data/output/output_stock_prices.csv", parse_dates=["Date"])

    # display the predictions on web
    st.write(f"{model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}")
    st.write(predictions_df)

    # make the figure using plotly
    fig = px.line(
        predictions_df,
        x="Date",
        y="Forecast",
        title=f"{model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}"
    )

    # plot it
    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Execute the whole pipeline
if __name__ == "__main__":

    main()