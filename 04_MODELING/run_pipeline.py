# Databricks notebook source
# MAGIC %md
# MAGIC ## XGBoost Hyperparameter Optimisation
# MAGIC 
# MAGIC **Objective**: This notebook's objective is train and optimise a XGBoost regression model
# MAGIC 
# MAGIC **Takeaways**: The key takeaways of this notebook are:

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

TARGET_VARIABLE = 'Weight'

# COMMAND ----------

# MAGIC %run ./extra_trees_hyperopt

# COMMAND ----------

# MAGIC %run ./xgboost_hyperopt

# COMMAND ----------

# MAGIC %run ./random_forest_hyperopt

# COMMAND ----------

# MAGIC %run ./lightgbm_hyperopt

# COMMAND ----------


