# Databricks notebook source
# MAGIC %md
# MAGIC ## Template
# MAGIC 
# MAGIC **Objective**: This notebook's objective is to explore the dataset and set the guidelines for modeling
# MAGIC 
# MAGIC **Takeaways**: The key takeaways of this notebook are:
# MAGIC 
# MAGIC - No missing values;
# MAGIC - No zero values;
# MAGIC - The target variable is not normal, but the features are near;
# MAGIC - The target variable has high correlation with the features;
# MAGIC - The categorical feature maybe will be of value (But dummy variables can reduce the model performance)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

df = spark.sql("select * from default.fish_cleaned").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Profilling

# COMMAND ----------

df.head()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Data visualisation

# COMMAND ----------

sns.pairplot(data=df)

# COMMAND ----------

fig, axs = plt.subplots(figsize=(12, 8))
sns.heatmap(data=df.corr(), annot=True)
plt.show()

# COMMAND ----------


