# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *
# make the dataset
PERIOD = '800d'
INTERVAL = '1d'

def load_production_model_params():

    client = MlflowClient()
    models_versions = []

    for mv in client.search_model_versions("name='{}'".format(model_config['REGISTER_MODEL_NAME_VAL'])):
        models_versions.append(dict(mv))

    current_prod_model = [x for x in models_versions if x['current_stage'] == 'Production'][0]
    prod_validation_model_params = mlflow.get_run(current_prod_model['run_id']).data.params

    # remove unsignificant params
    prod_validation_model_params_new = {}
    for key, value in prod_validation_model_params_new.items():
        if key in xgboost_hyperparameter_config.keys():
            prod_validation_model_params_new[key] = value

    return prod_validation_model_params_new, current_prod_model
        
# Execute the whole pipeline
if __name__ == "__main__":

    STOCK_NAME = 'BOVA11.SA'
    client = MlflowClient()

    logger.info("Starting the training pipeline..")

    # download the dataset and as raw
    # TODO: Stop downloading the dataset every time, just load it
    stock_df = make_dataset(STOCK_NAME, PERIOD, INTERVAL)

    # perform featurization
    stock_df_feat = build_features(stock_df, features_list)

    X_train=stock_df_feat.drop([model_config["TARGET_NAME"], "Date"], axis=1)
    y_train=stock_df_feat[model_config["TARGET_NAME"]]

    # load the production model parameters
    logger.debug("Loading the production model parameters..")
    prod_validation_model_params, current_prod_model = load_production_model_params()

    with mlflow.start_run(run_name="model_inference") as run:

        # use existing params
        xgboost_model = xgb.XGBRegressor(
            **prod_validation_model_params
        )

        # train the model
        xgboost_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            eval_metric=["rmse", "logloss"],
            verbose=0
        )

        learning_results = xgboost_model.evals_result()
      
        # Plotting the Learning Results
        fig2, axs = plt.subplots(1, 2, figsize=(12, 5))
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

        # ---- logging ----

        # log the parameters
        mlflow.log_params(prod_validation_model_params)#xgboost_model.get_params())
        mlflow.log_figure(fig2, "learning_curves.png")

        # get model signature
        model_signature = infer_signature(X_train, pd.DataFrame(y_train))

        # log the model to mlflow
        mlflow.xgboost.log_model(
            xgb_model=xgboost_model,
            artifact_path="xgboost_model",
            input_example=X_train.head(),
            signature=model_signature
        )

        # register the model
        logger.debug("Registering the model...")
        model_details = mlflow.register_model(
            model_uri = f"runs:/{run.info.run_id}/{run.info.run_name}",
            name = model_config['REGISTER_MODEL_NAME_INF']
        )

        # Archive the previous version
        client.transition_model_version_stage(
            name=model_config['REGISTER_MODEL_NAME_INF'],
            version=current_prod_model['version'],
            stage='Archived',
        )

        # transition the newest version
        client.transition_model_version_stage(
            name=model_config['REGISTER_MODEL_NAME_INF'],
            version=model_details.version,
            stage='Production',
        )

        # give model version a description
        client.update_model_version(
            name=model_config["REGISTER_MODEL_NAME_INF"],
            version=model_details.version,
            description=f"""This is the inference model for stock {STOCK_NAME}, trained based on the Hyeperparameters
            from the validation model version {current_prod_model['version']} \
            in Production."""
    )


    logger.info("\n\nTraining Pipeline was sucessful!\n")
