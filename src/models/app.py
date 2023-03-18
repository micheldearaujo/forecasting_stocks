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
    # data loading
    validation_report_df = pd.read_csv(os.path.join(OUTPUT_DATA_PATH, 'validation_stock_prices.csv'))


    st.write("""# Welcome to the Stock Forecaster!
    Here you can have a glance of how the future of your stocks can look like and *simulute* decisions based on that.
    Please keep in mind that this is just a educational tool and you should not perform financial operations based on that.
    """)

    st.sidebar.write("""### Choose your filters""")

    STOCK_NAME = st.sidebar.selectbox(
        "Which stock do you want to track?",
        ("BOVA11.SA", "ITUB4.SA", "VALE3.SA", "NFLX")
    )

    # get the historical starting date
    hist_start_date = st.sidebar.date_input(
        "From when do you want to the the historical prices?",
        dt.datetime.today().date() - dt.timedelta(days=2*model_config["FORECAST_HORIZON"])
    )

    # Definir as opções de cores para Forecasting e Historical
    historical_color = st.sidebar.color_picker('Escolha uma cor para Forecasting', '#FF5733')
    forecast_color = st.sidebar.color_picker('Escolha uma cor para Historical', '#5D6D7E')


    st.sidebar.write("""### Nerdzone""")
    st.sidebar.write("Here we have some technical details about the model and the data.")

    st.sidebar.write("#### Model")
    st.sidebar.write("Model hyperparameters: Work in Progress...")
    
    st.sidebar.write("#### Data")
    st.sidebar.write("Data source: Yahoo Finance")
    st.sidebar.write("Data processing: Work in Progress...")
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

    print(validation_report_df)

    # make the figure using plotly
    fig = px.line(
        full_df,
        x="Date",
        y="Price",
        color="Class",
        symbol="Class",
        title=f"{model_config['FORECAST_HORIZON']-4} days Forecast for {STOCK_NAME}",
        color_discrete_map={'Forecast': forecast_color, 'Historical': historical_color}
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



    # Opção para enviar feedback
    feedback = st.text_input("Envie seu feedback:")
    if feedback:
        st.write("Obrigado pelo seu feedback:", feedback[::-1])

# Execute the whole pipeline
if __name__ == "__main__":

    front_end()