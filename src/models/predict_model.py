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

    # make the dataset
    PERIOD = '800d'
    INTERVAL = '1d'
    features_list = ["day_of_month", "month", "quarter", "Close_lag_1"]
    client = MlflowClient()
    models_versions = []

    #STOCK_NAME = 'BOVA11.SA'  #str(input("Which stock do you want to track? "))
    STOCK_NAME = st.selectbox(
        "Which stock do you want to track?",
        ("BOVA11.SA")#, "ITUB4.SA", "VALE3.SA", "NFLX")
    )
    st.write("You selected:", STOCK_NAME)

    logger.info("Starting the Inference pipeline..")

    # load the raw dataset
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    print(stock_df.tail(2))

    # train the model
    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    # load the production model
    for mv in client.search_model_versions("name='{}'".format(model_config['REGISTER_MODEL_NAME_INF'])):
        models_versions.append(dict(mv))

    current_prod_model_info = [x for x in models_versions if x['current_stage'] == 'Production'][0]
    current_prod_model_uri = f"./mlruns/0/{current_prod_model_info['run_id']}/artifacts/xgboost_model"
    xgboost_model = mlflow.xgboost.load_model(model_uri=current_prod_model_uri)
    print(xgboost_model.get_params())
    
    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(model_config["FORECAST_HORIZON"], stock_df_feat, features_list)
    print(future_df)
    
    # Make predictions using the future dataframe and specified forecast horizon
    predictions_df = make_predict(
        model=xgboost_model,
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