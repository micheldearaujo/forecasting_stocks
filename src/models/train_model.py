# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *

def load_production_model_params(client: mlflow.tracking.client.MlflowClient, stock_name: str) -> tuple:

    # create empty list to store model versions
    models_versions = []

    # search for model versions
    for mv in client.search_model_versions("name='{}_{}'".format(model_config['REGISTER_MODEL_NAME_VAL'], stock_name)):
        models_versions.append(dict(mv))

    # get the prod model
    current_prod_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]
    prod_validation_model_params = mlflow.get_run(current_prod_model['run_id']).data.params

    prod_validation_model_params_new = {k: v for k, v in prod_validation_model_params.items() if k in xgboost_hyperparameter_config.keys()}

    return prod_validation_model_params_new, current_prod_model
        

def train_inference_model(X_train:pd.DataFrame, y_train: pd.Series, params: dict, stock_name: str) -> xgb.sklearn.XGBRegressor:
    
    # use existing params
    xgboost_model = xgb.XGBRegressor(
        eval_metric=["rmse", "logloss"],
        #**params
    )

    # train the model
    xgboost_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=0
    )

    # save as joblib
    dump(xgboost_model, f"./models/{stock_name}_{dt.datetime.today().date()}.joblib")

    return xgboost_model


def extract_learning_curves(model: xgb.sklearn.XGBRegressor, display: bool=False) -> matplotlib.figure.Figure:
    """Extracting the XGBoost Learning Curves.
    Can display the figure or not.

    Args:
        model (xgb.sklearn.XGBRegressor): Fit XGBoost model
        display (bool, optional): Display the figure. Defaults to False.

    Returns:
        _type_: Matplotlib figure
    """

    # extract the learning curves
    learning_results = model.evals_result()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
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

    if display:
        plt.show()
    
    return fig


def train_pipeline():

    # create the mlflow client
    client = MlflowClient()
    
    logger.debug("Loading the featurized dataset..")
    stock_df_feat_all = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'processed_stock_prices.csv'), parse_dates=["Date"])

    # iterate over the stocks
    for stock_name in stock_df_feat_all["Stock"].unique():

        # filter and drop the columns
        stock_df_feat = stock_df_feat_all[stock_df_feat_all["Stock"] == stock_name].drop("Stock", axis=1).copy()

        #logger.debug(f"Creating training dataset for stock {stock_name}..")
        logger.debug("Creating training dataset for stock %s..."%stock_name)
        
        X_train=stock_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1)
        y_train=stock_df_feat[model_config["TARGET_NAME"]]

        # load the production model parameters
        logger.debug("Loading the production model parameters..")
        prod_validation_model_params, current_prod_model = load_production_model_params(client, stock_name)
        mlflow.set_experiment(experiment_name="Training_Inference_Models")
        with mlflow.start_run(run_name=f"model_inference_{stock_name}") as run:

            logger.debug("Training the model..")
            xgboost_model = train_inference_model(X_train, y_train, prod_validation_model_params, stock_name)

            logger.debug("Plotting the learning curves..")
            fig = extract_learning_curves(xgboost_model)

            logger.debug("Logging the results..")
            # log the parameters
            mlflow.log_params(prod_validation_model_params)
            mlflow.log_figure(fig, f"learning_curves_{stock_name}.png")

            # get model signature
            model_signature = infer_signature(X_train, pd.DataFrame(y_train))

            # log the model to mlflow
            mlflow.xgboost.log_model(
                xgb_model=xgboost_model,
                artifact_path=f"{model_config['MODEL_NAME']}_{stock_name}",
                input_example=X_train.head(),
                signature=model_signature
            )

            # register the model
            logger.debug("Registering the model...")
            model_details = mlflow.register_model(
                model_uri = f"runs:/{run.info.run_id}/{run.info.run_name}",
                name = f"{model_config[f'REGISTER_MODEL_NAME_INF']}_{stock_name}"
            )

            logger.debug("Loading the current Inference Production model...")

            # Need to load the current prod inference model now, to archive it
            models_versions = []

            for mv in client.search_model_versions("name='{}_{}'".format(model_config[f'REGISTER_MODEL_NAME_INF'], stock_name)):
                models_versions.append(dict(mv))

            # Check if there is production model
            try:
                current_prod_inf_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]

                # Archive the previous version
                client.transition_model_version_stage(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=current_prod_inf_model['version'],
                    stage='Archived',
                )

                # transition the newest version
                client.transition_model_version_stage(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=model_details.version,
                    stage='Production',
                )

                # give model version a description
                client.update_model_version(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=model_details.version,
                    description=f"""This is the inference model for stock {STOCK_NAME}, trained based on the Hyperparameters
                    from the validation model version {current_prod_model['version']} \
                    in Production."""
                )

            except IndexError:

                # just set the new model as production
                client.transition_model_version_stage(
                    name=f"{model_config['REGISTER_MODEL_NAME_INF']}_{stock_name}",
                    version=model_details.version,
                    stage='Production',
                )


# Execute the whole pipeline
if __name__ == "__main__":
    logger.info("\nStarting the training pipeline...\n")

    train_pipeline()

    logger.info("Training Pipeline was sucessful!\n")

