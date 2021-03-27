# SparkStream Machine Learning Application 02 (Earthquake Prediction on Streaming Data)
### Dataflow Pipeline
![](pipeline.JPG)

### Summary

This project is created to predict Earthquake magnitude using Longitude, Latitude and Depth values on streaming data. 

### Task List

- [x] Set-up MongoDB account or install locally
- [x] Create quakes_etl.py
- [x] Create spark session and include mongo db connector package
```
# Create the spark session
spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('quakes_etl')\
    .config('spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .getOrCreate()
```
- [x] Read database.csv data and drop unnecessary column
```
# Load the dataset
df_load = spark.read.csv(r"database.csv", header=True)

# Remove all fields we don't need
lst_dropped_columns = ['Depth Error', 'Time', 'Depth Seismic Stations','Magnitude Error','Magnitude Seismic Stations','Azimuthal Gap', 'Horizontal Distance','Horizontal Error',
    'Root Mean Square','Source','Location Source','Magnitude Source','Status']

df_load = df_load.drop(*lst_dropped_columns)

```
- [x] Apply etl and create clean dataset for training
```
# Create a year field and add it to the df_load dataframe
df_load = df_load.withColumn('Year', year(to_timestamp('Date', 'dd/MM/yyyy')))

# Create the quakes freq dataframe form the year and count values
df_quake_freq = df_load.groupBy('Year').count().withColumnRenamed('count', 'Counts')

# Cast string fields to double types
df_load = df_load.withColumn('Latitude', df_load['Latitude'].cast(DoubleType()))\
    .withColumn('Longitude', df_load['Longitude'].cast(DoubleType()))\
    .withColumn('Depth', df_load['Depth'].cast(DoubleType()))\
    .withColumn('Magnitude', df_load['Magnitude'].cast(DoubleType()))

# Create avg and max magnitude fields and add to df_quake_freq
df_max = df_load.groupBy('Year').max('Magnitude').withColumnRenamed('max(Magnitude)', 'Max_Magnitude')
df_avg = df_load.groupBy('Year').avg('Magnitude').withColumnRenamed('avg(Magnitude)', 'Avg_Magnitude')

# Join the max and avg dfs to df_quake_freq
df_quake_freq = df_quake_freq.join(df_avg, ['Year']).join(df_max, ['Year'])

# Remove records with null values
df_load.dropna()
df_quake_freq.dropna()
```
- [x] Write data into mongodb
```
# Load df_load into mongodb
df_load.write.format('mongo')\
    .mode('overwrite')\
    .option('spark.mongodb.output.uri', 'mongodb+srv://username:password@cluster0.abcd.mongodb.net/quake.quakes').save()

# Load df_quake_freq into mongodb
df_quake_freq.write.format('mongo')\
    .mode('overwrite')\
    .option('spark.mongodb.output.uri', 'mongodb+srv://username:password@cluster0.abcd.mongodb.net/quake.quake_freq').save()
```
- [x] Create spark_kafka_nifi_stream_machine_learning.py file
- [x] Create spark session and load data from mongodb
```
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
```
- [x] Data pre processing before training
```
# Split data into train and test
df_train, df_test = df.randomSplit(weights = [0.80, 0.20], seed = 13)

#Arrange columns
df_training = df_train['Latitude', 'Longitude', 'Magnitude', 'Depth']
df_testing = df_test['Latitude', 'Longitude', 'Magnitude', 'Depth']

# Remove nulls from our datasets
df_training = df_training.dropna()
df_testing = df_testing.dropna()
```
- [x] Building machine learning model
```
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
```
- [x] Create prediction dataset to test algorithm
```
df_pred_results = pred_results['Latitude', 'Longitude', 'prediction']

# Rename the prediction field
df_pred_results = df_pred_results.withColumnRenamed('prediction', 'Pred_Magnitude')

# Add more columns
df_pred_results = df_pred_results \
    .withColumn('RMSE', lit(rmse))
```
- [x] Start apache nifi and create dataflow pipeline

![](nifi_pipeline.JPG)

- [x] Create kafka producer and consumer topics
- [x] Read query.csv data using GET file processor
- [x] Read streaming data from kafka with spark and define schema structure
```
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
```
- [x] Make prediction on streaming data
```
#Make prediction
pred_results_stream = model.transform(explode_df)
#Remove feature column
pred_results_stream_simplified = pred_results_stream.selectExpr("Latitude", "Longitude", "Depth", "prediction")
```
- [x] Serialize result and sink to kafka topic
```
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
```
- [x] Consume data from nifi and send it to Elasticsearch database
- [x] Creata visualization using Kibana

![](kibana.JPG)

### Code Description

quakes_etl.py is a file which we are preparing dataset before machine learning

spark_kafka_nifi_stream_machine_learning.py is created to make machine learning as well as making prediction on streaming data

Spark_ML_Streaming_Nifi_pipeline_01.xml is nifi dataflow template to simulate dataflow which we will use as near-real time data source to make prediction

### Running

1. Start zookeeper (Check kafka scripts)

```
zookeeper-server-start.bat config\zookeeper.properties (zookeeper-server-start.sh for linux)
```

2. Start kafka (Check kafka scripts)
```
kafka-server-start.bat config\server.properties  (kafka-server-start.sh for linux)
```

3. Execute below code first

```
spark-submit quakes.py
```

4. Start nifi

5. Start Elasticsearch

6. Start Kibana

7. Execute below code

```
spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.12:2.4.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1 spark_kafka_nifi_stream_machine_learning.py
```

8. Go Kibana server and create visualization
