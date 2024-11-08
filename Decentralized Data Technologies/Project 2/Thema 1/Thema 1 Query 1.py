# Databricks notebook source
#Importing everything required for spark sql

import pyspark
from pyspark.sql.functions import *

# COMMAND ----------

#Making the dataframe from the table that was created from the file

df_temp = spark.read.option("header",True).csv("/FileStore/tables/tempm-5.csv")

# COMMAND ----------

#Transform column date from String type to Date type

df_temp = df_temp.withColumn('Date',to_date('Date'))

# COMMAND ----------

#Transform column value from String type to Double type

df_temp = df_temp.withColumn("Value",col("Value").cast('double'))

# COMMAND ----------

#Extract minimum and maximum values for each day and produce a dataframe that contains them

df_min = df_temp.groupBy("Date").min("Value")
df_min = df_min.withColumn("Minimum",col("min(Value)"))
df_max = df_temp.groupBy("Date").max("Value")
df_max = df_max.withColumn("Maximum",col("max(Value)"))
df = (df_min.join(df_max, "Date").orderBy("Date")).drop("min(Value)","max(Value)")

# COMMAND ----------

#Find the dates with minimum value over 18 and maximum below 22 and count how many there are

df_q =  df.filter(df["Minimum"] >= 18).filter(df["Maximum"] <= 22)
q_count = df_q.count()

# COMMAND ----------

#Print the results and the count

print("Dataframe for dates with temperature between 18 and 22 celcius: ")
df_q.printSchema()
df_q.show()

print("Number of days with temperature between 18 and 22 celcius: ",q_count)
