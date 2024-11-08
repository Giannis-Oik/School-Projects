# Databricks notebook source
#Importing everything required for spark sql

import pyspark
from pyspark.sql import Window
from pyspark.sql.functions import *

# COMMAND ----------

#Making the dataframes from the tables that were created from each file

df_agn = spark.read.option("header",True).csv("/FileStore/tables/agn_us.txt")
df_ainv = spark.read.option("header",True).csv("/FileStore/tables/ainv_us.txt")
df_ale = spark.read.option("header",True).csv("/FileStore/tables/ale_us.txt")

# COMMAND ----------

#Transform column date from String type to Date type

df_agn = df_agn.withColumn('Date',to_date('Date'))
df_ainv = df_ainv.withColumn('Date',to_date('Date'))
df_ale = df_ale.withColumn('Date',to_date('Date'))

# COMMAND ----------

#Transforming each column that has double type values from String type to Double type

columns = ['Open','High','Low','Close']
for x in columns:
    df_agn = df_agn.withColumn(x,col(x).cast('double'))
    df_ainv = df_ainv.withColumn(x,col(x).cast('double'))
    df_ale = df_ale.withColumn(x,col(x).cast('double'))

# COMMAND ----------

#Transform each column that has integer type values from String type to Int type

columns = ['Volume','OpenInt']
for x in columns:
    df_agn = df_agn.withColumn(x,col(x).cast('int'))
    df_ainv = df_ainv.withColumn(x,col(x).cast('int'))
    df_ale = df_ale.withColumn(x,col(x).cast('int'))

# COMMAND ----------

#Create a new column where the values are the years extracted from the Date column

df_agn = df_agn.withColumn("Year",year("Date"))
df_ainv = df_ainv.withColumn("Year",year("Date"))
df_ale = df_ale.withColumn("Year",year("Date"))

# COMMAND ----------

#Query that create structs with Open,Date values for each entry in the original dataframe 
#Then finds the maximum or minimum struct by Open and Close 
#And then through join returns a dataframe that has the maximum Open, minimum Close and Year values 

df_agn_open = df_agn.withColumn('Open_Year_struct', struct(df_agn.Open, df_agn.Year))
df_agn_close = df_agn.withColumn('Close_Year_struct', struct(df_agn.Close, df_agn.Year))
max_df_open = df_agn_open.agg(max('Open_Year_struct').alias('Max_Open'))
min_df_close = df_agn_close.agg(min('Close_Year_struct').alias('Min_Close'))
q_agn = (max_df_open.withColumn('Max Year', max_df_open.Max_Open.Year).withColumn('Max Open', max_df_open.Max_Open.Open).drop('Max_Open')).join(min_df_close.withColumn('Min Year', min_df_close.Min_Close.Year).withColumn('Min Close', min_df_close.Min_Close.Close).drop('Min_Close'))

df_ainv_open = df_ainv.withColumn('Open_Year_struct', struct(df_ainv.Open, df_ainv.Year))
df_ainv_close = df_ainv.withColumn('Close_Year_struct', struct(df_ainv.Close, df_ainv.Year))
max_df_open = df_ainv_open.agg(max('Open_Year_struct').alias('Max_Open'))
min_df_close = df_ainv_close.agg(min('Close_Year_struct').alias('Min_Close'))
q_ainv = (max_df_open.withColumn('Max Year', max_df_open.Max_Open.Year).withColumn('Max Open', max_df_open.Max_Open.Open).drop('Max_Open')).join(min_df_close.withColumn('Min Year', min_df_close.Min_Close.Year).withColumn('Min Close', min_df_close.Min_Close.Close).drop('Min_Close'))

df_ale_open = df_ale.withColumn('Open_Year_struct', struct(df_ale.Open, df_ale.Year))
df_ale_close = df_ale.withColumn('Close_Year_struct', struct(df_ale.Close, df_ale.Year))
max_df_open = df_ale_open.agg(max('Open_Year_struct').alias('Max_Open'))
min_df_close = df_ale_close.agg(min('Close_Year_struct').alias('Min_Close'))
q_ale = (max_df_open.withColumn('Max Year', max_df_open.Max_Open.Year).withColumn('Max Open', max_df_open.Max_Open.Open).drop('Max_Open')).join(min_df_close.withColumn('Min Year', min_df_close.Min_Close.Year).withColumn('Min Close', min_df_close.Min_Close.Close).drop('Min_Close'))

# COMMAND ----------

#Print the schema for each dataframe and show it, where each dataframe contains the maximum and minimum values for Open and the Year each one was achieved

print("Max and min Value for AGN: ")
q_agn.printSchema()
q_agn.show()

print("Max and min Value for AINV: ")
q_ainv.printSchema()
q_ainv.show()

print("Max and min Value for ALE: ")
q_ale.printSchema()
q_ale.show()
