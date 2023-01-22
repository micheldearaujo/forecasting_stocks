# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

from config import *


def build_features(raw_df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    This function creates the features for the dataset to be consumed by the
    model
    
    :param raw_df: Raw Pandas DataFrame to create the features of
    :param features_list: The list of features to create

    :return: Pandas DataFrame with the new features
    """

    stock_df_featurized = raw_df.copy()
    for feature in features_list:
        
        # Create "Time" features]
        if feature == "day_of_month":
            stock_df_featurized['day_of_month'] = stock_df_featurized["Date"].apply(lambda x: x.day)
        elif feature == "month":
            stock_df_featurized['month'] = stock_df_featurized['Date'].apply(lambda x: x.month)
        elif feature == "quarter":
            stock_df_featurized['quarter'] = stock_df_featurized['Date'].apply(lambda x: x.quarter)

    # Create "Lag" features
    # The lag 1 feature will become the main regressor, and the regular "Close" will become the target.
    # As we saw that the lag 1 holds the most aucorrelation, it is reasonable to use it as the main regressor.
        elif feature == "Close_lag_1":
            stock_df_featurized['Close_lag_1'] = stock_df_featurized['Close'].shift()


    # Drop nan values because of the shift
    stock_df_featurized = stock_df_featurized.dropna()
    # Drop the Date column
    #stock_df_featurized = stock_df_featurized.drop("Date", axis=1)
    # Save the dataset
    stock_df_featurized.to_csv("./data/processed/processed_stock_prices.csv", index=False)

    return stock_df_featurized


def ts_train_test_split(data, target, test_size):
    train_df = data.iloc[:-test_size, :]
    test_df = data.iloc[-test_size:, :]
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


def visualize_validation_results(pred_df, model_mape, model_rmse):

    """
    Creates visualizations of the model training and validation
    """

    # Transform predictions into dataframe
    

    fig, axs = plt.subplots(figsize=(12, 5))
    # Plot the Actuals
    sns.lineplot(
        data=pred_df,
        x="Date",
        y="Actual",
        label="Testing values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Actual",
        ax=axs,
        size="Actual",
        sizes=(80, 80), legend=False
    )

    # Plot the Forecasts
    sns.lineplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        label="Forecast values",
        ax=axs
    )
    sns.scatterplot(
        data=pred_df,
        x="Date",
        y="Forecast",
        ax=axs,
        size="Forecast",
        sizes=(80, 80), legend=False
    )

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']} days Forecast for {STOCK_NAME}\nMAPE: {round(model_mape*100, 2)}% | RMSE: R${model_rmse}")
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    plt.savefig(f"./XGBoost_predictions_{dt.datetime.now()}.png")


def validate_model(model, X_val):
    """
    Perform predictions to validate the model
    
    :param model: The Fitted model
    :param X_val: Validation Features
    :Param y_val: Validation Target
    """

    prediction = model.predict(X_val)

    return prediction


def train_model(X_train, y_train, random_state=42):
    """
    Trains a XGBoost model for Forecasting
    
    :param X_train: Training Features
    :param y_train: Training Target

    :return: Fitted model
    """
    # create the model
    xgboost_model = xgb.XGBRegressor(
        random_state=random_state,
        )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train, 
        )

    return xgboost_model


def make_predictions(X, y, forecast_horizon):
    """
    Iterate over forecast horizon performing
    stepwise iterative predictions
    """

    # Create empty list for storing each prediction
    predictions = []
    actuals = []
    dates = []


    # Iterate over the dataset to perform predictions over the forecast horizon, one by one.
    # So we need to start at training = training until the total forecast horizon, then, perform the next step
    # After forecasting the next step, we need to append the new line to the training dataset and so on

    for day in range(forecast_horizon, 0, -1):
        print(day)
        # update the training and testing sets
        X_train = X.iloc[:-day, :]
        y_train = y.iloc[:-day]
        print("Training until", X_train["Date"].max())
        if day == 1:
            X_test = X.iloc[-day:,:]
            y_test = y.iloc[-day:]
        else:
            X_test = X.iloc[-day:-day+1,:]
            y_test = y.iloc[-day:-day+1]
        print("Testing for day: ", X_test)
        #print(X_test)

        # train the model
        xgboost_model = train_model(X_train.drop("Date", axis=1), y_train)

        # validade the model
        prediction = validate_model(xgboost_model, X_test.drop("Date", axis=1))
        predictions.append(prediction[0])
        actuals.append(y_test.values[0])
        dates.append(X_test["Date"].max())

    
    # Calculate the resulting metric
    model_mape = round(mean_absolute_percentage_error(actuals, predictions), 4)
    print(model_mape)
    model_rmse = round(np.sqrt(mean_squared_error(actuals, predictions)), 2)
 
    pred_df = pd.DataFrame(list(zip(dates, actuals, predictions)), columns=["Date", 'Actual', 'Forecast'])
    visualize_validation_results(pred_df, model_mape, model_rmse)






