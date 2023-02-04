# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0,'.')

from src.utils import *

PERIOD = '800d'
INTERVAL = '1d'

STOCK_NAME = 'BOVA11.SA'

# create a fake dataset
dates_list = ['2022-01-01', '2022-01-02', '2022-01-03']
dates_list = [pd.to_datetime(date) for date in dates_list]
prices_list = [np.random.rand()*10, 100, 4571.54]
prices_list = [float(price) for price in prices_list]
day_of_months_list = [1., 2., 30.]
months_list = [1., 6., 12.]
quarters_list = [1., 2., 4.]
close_lags_list = prices_list
TEST_FORECAST_HORIZON = 1

test_stock_df = pd.DataFrame(
    {
        "Date": dates_list,
        "Close": prices_list
    }
)

test_stock_feat_df = pd.DataFrame(
    {
        "Date": dates_list,
        "Close": prices_list,
        "day_of_month": day_of_months_list,
        "month": months_list,
        "quarter": quarters_list,
        "Close_lag_1": close_lags_list
    }
)

test_predictions_df = pd.DataFrame(
    {
        "Date": dates_list,
        "Actual": prices_list,
        "Forecast": prices_list
    }
)


# def test_make_dataset():
#     """
#     tests if the make_dataset() is download
#     and saving the file in the correct format
#     """

#     # call the make_dataset function
#     stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

#     assert test_stock_df.columns.all() == stock_price_df.columns.all()
#     assert isinstance(stock_price_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(stock_price_df["Close"].dtype, type(np.dtype("float64")))
#     assert stock_price_df.shape[0] >= int(PERIOD[:-1])


def test_make_dataset_columns():
    """
    tests if the make_dataset() is download
    and saving the file in the correct format
    """

    # call the make_dataset function
    stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)
    
    # assert the amount of columns and column's orders
    assert test_stock_df.columns.all() == stock_price_df.columns.all()


def test_make_dataset_types():
    """
    tests if the make_dataset() is download
    and saving the file in the correct format
    """

    # call the make_dataset function
    stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # assert the columns data types
    assert isinstance(stock_price_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(stock_price_df["Close"].dtype, type(np.dtype("float64")))


def test_make_dataset_size():
    """
    tests if the make_dataset() is download
    and saving the file in the correct format
    """

    # call the make_dataset function
    stock_price_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # asser the amount of days
    assert stock_price_df.shape[0] >= int(PERIOD[:-1])


def test_build_features_columns():
    """
    tests if the build() is download
    and saving the file in the correct format
    """
    # load the output file to test

    stock_df_feat = build_features(test_stock_df, features_list)

    assert test_stock_feat_df.columns.all() == stock_df_feat.columns.all()


def test_build_features_types():
    """
    tests if the build() is download
    and saving the file in the correct format
    """
    # load the output file to test

    stock_df_feat = build_features(test_stock_df, features_list)


    assert isinstance(stock_df_feat["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(stock_df_feat["Close"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["day_of_month"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["month"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["quarter"].dtype, type(np.dtype("float64")))
    assert isinstance(stock_df_feat["Close_lag_1"].dtype, type(np.dtype("float64")))
    

def test_build_features_size():
    """
    tests if the build() is download
    and saving the file in the correct format
    """

    # load the output file to test
    stock_df_feat = build_features(test_stock_df, features_list)
    
    assert stock_df_feat.shape[0] == test_stock_df.shape[0] - 1  # because of the shift(1)


# def test_build_features():
#     """
#     tests if the build() is download
#     and saving the file in the correct format
#     """
#     # load the output file to test

#     stock_df_feat = build_features(test_stock_df, features_list)

#     assert test_stock_feat_df.columns.all() == stock_df_feat.columns.all()
#     assert isinstance(stock_df_feat["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(stock_df_feat["Close"].dtype, type(np.dtype("float64")))
#     assert isinstance(stock_df_feat["day_of_month"].dtype, type(np.dtype("float64")))
#     assert isinstance(stock_df_feat["month"].dtype, type(np.dtype("float64")))
#     assert isinstance(stock_df_feat["quarter"].dtype, type(np.dtype("float64")))
#     assert isinstance(stock_df_feat["Close_lag_1"].dtype, type(np.dtype("float64")))



# def test_ts_train_test_split():

#     returned_array = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"], TEST_FORECAST_HORIZON)

#     assert len(returned_array) == 4
#     X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

#     assert test_stock_feat_df.columns.all() == X_train.columns.all()
#     assert test_stock_feat_df.columns.all() == X_test.columns.all()
    
#     assert isinstance(X_train["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(X_train["day_of_month"].dtype, type(np.dtype("float64")))
#     assert isinstance(X_train["month"].dtype, type(np.dtype("float64")))
#     assert isinstance(X_train["quarter"].dtype, type(np.dtype("float64")))
#     assert isinstance(X_train["Close_lag_1"].dtype, type(np.dtype("float64")))

#     assert isinstance(X_test["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(X_test["day_of_month"].dtype, type(np.dtype("float64")))
#     assert isinstance(X_test["month"].dtype, type(np.dtype("float64")))
#     assert isinstance(X_test["quarter"].dtype, type(np.dtype("float64")))
#     assert isinstance(X_test["Close_lag_1"].dtype, type(np.dtype("float64")))

#     assert isinstance(y_train.dtype, type(np.dtype("float64")))
#     assert isinstance(y_test.dtype, type(np.dtype("float64")))



def test_ts_train_test_split_array_size():

    returned_array = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"], TEST_FORECAST_HORIZON)

    assert len(returned_array) == 4


def test_ts_train_test_split_columns():

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert test_stock_feat_df.columns.all() == X_train.columns.all()
    assert test_stock_feat_df.columns.all() == X_test.columns.all()


def test_ts_train_test_split_train_types():

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)
    
    assert isinstance(X_train["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(X_train["day_of_month"].dtype, type(np.dtype("float64")))
    assert isinstance(X_train["month"].dtype, type(np.dtype("float64")))
    assert isinstance(X_train["quarter"].dtype, type(np.dtype("float64")))
    assert isinstance(X_train["Close_lag_1"].dtype, type(np.dtype("float64")))
    assert isinstance(y_train.dtype, type(np.dtype("float64")))


# def test_train_model():
#     # train the model
#     xgboost_model = train_model(test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1), test_stock_feat_df[model_config["TARGET_NAME"]])

#     assert isinstance(xgboost_model, xgb.sklearn.XGBRegressor)
#     assert list(xgboost_model.feature_names_in_) == list(test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1).columns)

def test_train_model_types():
    # train the model
    xgboost_model = train_model(test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1), test_stock_feat_df[model_config["TARGET_NAME"]])

    assert isinstance(xgboost_model, xgb.sklearn.XGBRegressor)


def test_train_model_features():
    # train the model
    xgboost_model = train_model(test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1), test_stock_feat_df[model_config["TARGET_NAME"]])

    assert list(xgboost_model.feature_names_in_) == list(test_stock_feat_df.drop([model_config["TARGET_NAME"], "Date"], axis=1).columns)


# def test_validate_model():

#     predictions_df = validate_model(
#         X=test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1),
#         y=test_stock_feat_df[model_config["TARGET_NAME"]],
#         forecast_horizon=TEST_FORECAST_HORIZON
#     )

#     assert test_predictions_df.columns.all() == predictions_df.columns.all()
#     assert isinstance(predictions_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(predictions_df["Actual"].dtype, type(np.dtype("float64")))
#     assert isinstance(predictions_df["Forecast"].dtype, type(np.dtype("float64")))
#     assert predictions_df.shape[0] == TEST_FORECAST_HORIZON


def test_validate_model_stepwise_columns():

    predictions_df = validate_model_stepwise(
        X=test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1),
        y=test_stock_feat_df[model_config["TARGET_NAME"]],
        forecast_horizon=TEST_FORECAST_HORIZON
    )

    assert test_predictions_df.columns.all() == predictions_df.columns.all()


def test_validate_model_stepwise_types():

    predictions_df = validate_model_stepwise(
        X=test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1),
        y=test_stock_feat_df[model_config["TARGET_NAME"]],
        forecast_horizon=TEST_FORECAST_HORIZON
    )

    assert isinstance(predictions_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(predictions_df["Actual"].dtype, type(np.dtype("float64")))
    assert isinstance(predictions_df["Forecast"].dtype, type(np.dtype("float64")))


def test_validate_model_stepwise_size():

    predictions_df = validate_model_stepwise(
        X=test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1),
        y=test_stock_feat_df[model_config["TARGET_NAME"]],
        forecast_horizon=TEST_FORECAST_HORIZON
    )

    assert predictions_df.shape[0] == TEST_FORECAST_HORIZON



# def test_make_future_df():

#     # Create the future dataframe using the make_future_df function
#     future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

#     X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

#     assert X_train.columns.all() == future_df.columns.all()
#     assert isinstance(future_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(future_df["day_of_month"].dtype, type(np.dtype("float64")))
#     assert isinstance(future_df["month"].dtype, type(np.dtype("float64")))
#     assert isinstance(future_df["quarter"].dtype, type(np.dtype("float64")))
#     assert isinstance(future_df["Close_lag_1"].dtype, type(np.dtype("float64")))
#     assert future_df.shape[0] == TEST_FORECAST_HORIZON


def test_make_future_df_columns():

    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert X_train.columns.all() == future_df.columns.all()


def test_make_future_df_types():

    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert isinstance(future_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(future_df["day_of_month"].dtype, type(np.dtype("float64")))
    assert isinstance(future_df["month"].dtype, type(np.dtype("float64")))
    assert isinstance(future_df["quarter"].dtype, type(np.dtype("float64")))
    assert isinstance(future_df["Close_lag_1"].dtype, type(np.dtype("float64")))


def test_make_future_df_size():

    # Create the future dataframe using the make_future_df function
    future_df = make_future_df(TEST_FORECAST_HORIZON, test_stock_feat_df, features_list)

    X_train, X_test, y_train, y_test = ts_train_test_split(test_stock_feat_df, model_config["TARGET_NAME"],TEST_FORECAST_HORIZON)

    assert future_df.shape[0] == TEST_FORECAST_HORIZON
    

# def test_make_predict():

#     test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1).copy()
#     predictions_df = make_predict(
#         forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
#         future_df=test_inference_df
#     )
    
#     assert test_predictions_df.columns.all() == predictions_df.columns.all()
#     assert isinstance(predictions_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
#     assert isinstance(predictions_df["Forecast"].dtype, type(np.dtype("float64")))
#     assert predictions_df.shape[0] == TEST_FORECAST_HORIZON*test_inference_df.shape[0]

def test_make_predict_columns():

    test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1).copy()

    predictions_df = make_predict(
        forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
        future_df=test_inference_df
    )
    
    assert test_predictions_df.columns.all() == predictions_df.columns.all()


def test_make_predict_types():

    test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1).copy()
    predictions_df = make_predict(
        forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
        future_df=test_inference_df
    )

    assert isinstance(predictions_df["Date"].dtype, type(np.dtype("datetime64[ns]")))
    assert isinstance(predictions_df["Forecast"].dtype, type(np.dtype("float64")))


def test_make_predict_size():

    test_inference_df = test_stock_feat_df.drop([model_config["TARGET_NAME"]], axis=1).copy()
    predictions_df = make_predict(
        forecast_horizon=TEST_FORECAST_HORIZON*test_inference_df.shape[0],
        future_df=test_inference_df
    )
    
    assert predictions_df.shape[0] == TEST_FORECAST_HORIZON*test_inference_df.shape[0]




    

#test_make_dataset()
#test_build_features()
#test_ts_train_test_split()
#test_train_model()
#test_validate_model()
#test_make_future_df()
#test_make_predict()