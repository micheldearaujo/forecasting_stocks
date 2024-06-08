# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
from xgboost import plot_importance
import warnings
import yaml

warnings.filterwarnings("ignore")

with open("src/configuration/logging_config.yaml", 'r') as f:  

    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)
        

def train_model(X_train, y_train, params):
    """Treina um modelo XGBoost para regressão."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain)
    return model

def predict(model, X_test):
    """Realiza previsões com o modelo XGBoost treinado."""
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    return y_pred


def train_inference_model(X_train:pd.DataFrame, y_train: pd.Series, stock_name: str) -> xgb.sklearn.XGBRegressor:
    """
    Trains the XGBoost model with the full dataset to perform out-of-sample inference.
    """
    
    # use existing params
    xgboost_model = xgb.XGBRegressor(
        eval_metric=["rmse", "logloss"],
        n_estimators=40,
        max_depth=11
    )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=20
    )

    return xgboost_model


def extract_learning_curves(model: xgb.sklearn.XGBRegressor, display: bool=False) -> matplotlib.figure.Figure:
    """
    Extracting the XGBoost Learning Curves.
    Can display the figure or not.

    Args:
        model (xgb.sklearn.XGBRegressor): Fitted XGBoost model
        display (bool, optional): Display the figure. Defaults to False.

    Returns:
        matplotlib.figure.Figure: Learning curves figure
    """

    # extract the learning curves
    learning_results = model.evals_result()

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    plt.suptitle("XGBoost Learning Curves")
    axs[0].plot(learning_results['validation_0']['rmse'], label='Training')
    axs[0].set_title("RMSE Metric")
    axs[0].set_ylabel("RMSE")
    axs[0].set_xlabel("Iterations")
    axs[0].legend()

    axs[1].plot(learning_results['validation_0']['logloss'], label='Training')
    axs[1].set_title("Logloss Metric")
    axs[1].set_ylabel("Logloss")
    axs[1].set_xlabel("Iterations")
    axs[1].legend()

    fig2, axs2 = plt.subplots(figsize=(6, 3))
    plot_importance(model, ax=axs2, importance_type='gain')
    plt.close()
    

    if display:
        plt.show()
        
    
    return fig, fig2


def train_pipeline():

    client = MlflowClient()
    
    logger.debug("Loading the featurized dataset..")
    all_ticker_symbols_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    for ticker_symbol in all_ticker_symbols_df["Stock"].unique():
        logger.info("Creating training dataset for Ticker Symbol [%s...]"%ticker_symbol)

        ticker_df_feat = all_ticker_symbols_df[all_ticker_symbols_df["Stock"] == ticker_symbol].drop("Stock", axis=1).copy()

        X_train=ticker_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1)
        y_train=ticker_df_feat[model_config["TARGET_NAME"]]

        logger.debug("Training the model..")
        xgboost_model = train_inference_model(X_train, y_train, ticker_symbol)

        logger.debug("Plotting the learning curves..")
        # learning_curves_fig , feat_importance_fig = extract_learning_curves(xgboost_model)

        # Saves the model
        os.makedirs(MODELS_PATH, exist_ok=True)
        xgboost_model.save_model(f"{MODELS_PATH}/XGB_{ticker_symbol}.json")


# Execute the whole pipeline
if __name__ == "__main__":
    logger.info("Starting the training pipeline...")

    train_pipeline()

    logger.info("Training Pipeline was sucessful!\n")

