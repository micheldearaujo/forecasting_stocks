# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------

import sys

sys.path.insert(0,'.')

import yaml

from src.config import *


with open("src/configuration/hyperparams.yaml", 'r') as f:  

    model_config = yaml.safe_load(f.read())

with open("src/configuration/logging_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

# Organizing all parameter distributions into one dictionary
# param_space_dict = { 
#     'ExtraTreesRegressor': model_config['PARAM_SPACES']['et'],
#     'XGBRegressor': model_config['PARAM_SPACES']['xgb'],
#     'LightGBM': model_config['PARAM_SPACES']['lgb']
# }

def write_dataset_to_file(df: pd.DataFrame, dir_path: str, file_name: str) -> None:
    """Saves any DataFrame to a CSV file in the specified directory."""

    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(os.path.join(dir_path, f'{file_name}.csv'), index=False)


def weekend_adj_forecast_horizon(original_forecast_horizon, weekend_days_per_week):

    adjusted_forecast_horizon = original_forecast_horizon - int(original_forecast_horizon * weekend_days_per_week / 7)

    return adjusted_forecast_horizon


def visualize_validation_results(pred_df: pd.DataFrame, model_mape: float, model_mae: float, model_wape: float, stock_name: str):
    """
    Creates visualizations of the model validation

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        model_mape: The validation MAPE
        model_mae: The validation MAE
        model_wape: The validation WAPE

    Returns:
        None
    """

    logger.info("Vizualizing the results...")

    fig, axs = plt.subplots(figsize=(6, 3))

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

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']} days Forecast for {stock_name}\nMAPE: {round(model_mape*100, 2)}% | MAE: R${model_mae} | WAPE: {model_wape}")
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    try:
        plt.savefig(f"./reports/figures/XGBoost_predictions_{dt.datetime.now().date()}_{stock_name}.png")

    except FileNotFoundError:
        logger.warning("Forecast Figure not Saved!")

    #plt.show()
    return fig


def visualize_forecast(pred_df: pd.DataFrame, historical_df: pd.DataFrame, stock_name: str):
    """
    Creates visualizations of the model forecast

    Paramters:
        pred_df: DataFrame with true values, predictions and the date column
        historical_df: DataFrame with historical values

    Returns:
        None
    """

    logger.info("Vizualizing the results...")

    fig, axs = plt.subplots(figsize=(12, 5), dpi = 2000)
    # Plot the Actuals
    sns.lineplot(
        data=historical_df,
        x="Date",
        y="Close",
        label="Historical values",
        ax=axs
    )
    sns.scatterplot(
        data=historical_df,
        x="Date",
        y="Close",
        ax=axs,
        size="Close",
        sizes=(80, 80),
        legend=False
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
        sizes=(80, 80),
        legend=False
    )

    axs.set_title(f"Default XGBoost {model_config['FORECAST_HORIZON']-4} days Forecast for {stock_name}")
    axs.set_xlabel("Date")
    axs.set_ylabel("R$")

    #plt.show()
    return fig
