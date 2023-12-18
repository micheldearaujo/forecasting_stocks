# Run the whole project
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,'.')

from src.utils import *

from src.features.build_features import build_features
from src.data.make_dataset import make_dataset
from src.models.test_model import model_testing_pipeline
from src.models.train_model import extract_learning_curves, train_pipeline
from src.models.predict_model import inference_pipeline
from src.models.model_utils import cd_pipeline
from src.config import features_list

# logger.info("Downloading the raw dataset...")
# stock_df = make_dataset(stocks_list, PERIOD, INTERVAL)
# logger.info("Finished downloading the raw dataset!")

# logger.info("Featurizing the dataset...")
# stock_df_feat = build_features(stock_df, features_list)
# logger.info("Finished featurizing the dataset!")

logger.info("Starting the Model Testing pipeline...")
model_testing_pipeline()
logger.info("Model Testing Pipeline was sucessful!")

logger.info("Starting the training pipeline...")
train_pipeline()
logger.info("Training Pipeline was sucessful!\n")

logger.info("Starting the Inference pipeline...")
inference_pipeline()
logger.info("Inference Pipeline was successful!\n")