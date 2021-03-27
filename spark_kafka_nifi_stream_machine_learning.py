import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, VectorSizeHint
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import json

# Create spark session
spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('quakes_ml')\
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

"""
Data Pre-processing
"""
# Load quakes data from mongodb
df = spark.read.format('mongo')\
    .option('spark.mongodb.input.uri', 'mongodb+srv://username:password@cluster0.abcd.mongodb.net/quake.quakes').load()

# Split data into train and test
df_train, df_test = df.randomSplit(weights = [0.80, 0.20], seed = 13)

#Arrange columns
df_training = df_train['Latitude', 'Longitude', 'Magnitude', 'Depth']
df_testing = df_test['Latitude', 'Longitude', 'Magnitude', 'Depth']

# Remove nulls from our datasets
df_training = df_training.dropna()
df_testing = df_testing.dropna()

"""
Building the machine learning model
"""
# Create feature vector
assembler = VectorAssembler(inputCols=['Latitude', 'Longitude', 'Depth'], outputCol='features').setHandleInvalid("skip")

# Create the model
model_reg = RandomForestRegressor(featuresCol='features', labelCol='Magnitude')

# Chain assembler and model into a pipleine
pipeline = Pipeline(stages=[assembler, model_reg])

# Train the Model
model = pipeline.fit(df_training)

# Make the prediction
pred_results = model.transform(df_testing)

# Evaluate model
evaluator = RegressionEvaluator(labelCol='Magnitude', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(pred_results)

"""
Create the prediction dataset
"""
df_pred_results = pred_results['Latitude', 'Longitude', 'prediction']

# Rename the prediction field
df_pred_results = df_pred_results.withColumnRenamed('prediction', 'Pred_Magnitude')

# Add more columns
df_pred_results = df_pred_results \
    .withColumn('RMSE', lit(rmse))
print(df_pred_results.show(5))

print('INFO: Job ran successfully')
print('')

"""
Streaming Part
"""
#Read from kafka topic "quake"
kafka_df = spark \
    .readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', "localhost:9092") \
    .option("startingOffsets", "earliest") \
    .option('subscribe', 'quake') \
    .load()
#Define schema
schema = StructType([StructField('Latitude', DoubleType()),\
                     StructField('Longitude', DoubleType()),\
                     StructField('Depth', DoubleType())])
#Print schema to review
kafka_df.printSchema()
#Deserialize json object and apply schema
value_df = kafka_df.select(from_json(col("value").cast("string"),schema).alias("value"))
#Print schema to review
value_df.printSchema()
#Flatten dataframe
explode_df = value_df.selectExpr("value.Latitude","value.Longitude", "value.Depth")
#Print schema to review
explode_df.printSchema()
#Make prediction
pred_results_stream = model.transform(explode_df)
#Remove feature column
pred_results_stream_simplified = pred_results_stream.selectExpr("Latitude", "Longitude", "Depth", "prediction")

kafka_df = pred_results_stream_simplified.select("*")

kafka_df = kafka_df.selectExpr("cast(Latitude as string) Latitude","Longitude", "Depth", "prediction")

kafka_target_df = kafka_df.selectExpr("Latitude as key",
                                             "to_json(struct(*)) as value")

kafka_target_df.printSchema()

nifi_query = kafka_target_df \
        .writeStream \
        .queryName("Notification Writer") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "quake2") \
        .outputMode("append") \
        .option("checkpointLocation", "chk-point-dir") \
        .start()

nifi_query.awaitTermination()

## Below command used to preview results on the console before inserting data to database
#Sink result to console
# window_query = pred_results_stream_simplified.writeStream \
#      .format("console") \
#      .outputMode("append") \
#      .trigger(processingTime="10 seconds") \
#      .start()
#
# window_query.awaitTermination()
