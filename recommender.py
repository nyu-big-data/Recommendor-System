#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit recommender.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
# Use time to calculate model fitting time
import time

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode, col

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    # ratings = spark.read.csv('ratings.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    ratings = spark.read.csv('ratings-small.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()

    ratings.createOrReplaceTempView('ratings')

    # reading from the partitioned training dataset
    training = spark.read.csv('train-small-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    # training = spark.read.csv('train-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    training = training.drop("timestamp")
    training.createOrReplaceTempView('training')

    # Construct a query to get the top 100 movies with highest ratings
    print('Getting top 100 movies with highest ratings')

    start_time = time.time()

    predicted_ratings = spark.sql('SELECT movieId, (SUM(rating)/COUNT(rating)) AS predicted_rating FROM ratings GROUP BY movieId HAVING COUNT(rating) > 0 ORDER BY predicted_rating DESC LIMIT 100')
    
    end_time = time.time()
    print("Total popularity baseline model query execution time: {} seconds".format(end_time - start_time))

    # Print the predicted ratings to the console
    predicted_ratings.show()
    predicted_ratings.createOrReplaceTempView('predicted_ratings')

    # reading test data from the partitioned file
    test = spark.read.csv('test-small-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    # test = spark.read.csv('test-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    test = test.drop("timestamp")
    test.createOrReplaceTempView('test')

    # joining 2 tables and leaving only the ratings from each table to be compared
    scoreAndLabels = spark.sql('SELECT rating, predicted_rating FROM predicted_ratings LEFT JOIN test ON predicted_ratings.movieId = test.movieId').na.drop()


    # Instantiate regression metrics to compare predicted and actual ratings
    metrics = RegressionMetrics(scoreAndLabels.rdd)

    print("With popularity baseline model: ")

    # Root mean squared error
    print("Popularity baseline model RMSE = %s" % metrics.rootMeanSquaredError)

    # R-squared
    print("popularity baseline model R-squared = %s" % metrics.r2)

    print("---------------------------------------------------")

    print("with ALS model: ")

    # training
    start_time = time.time()

    als = ALS(rank=200, maxIter=15, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",  nonnegative=True ,coldStartStrategy="drop")
    model = als.fit(training)

    end_time = time.time()
    print("Total ALS model training time: {} seconds".format(end_time - start_time))


    # Tune the hyperparameters with the validation data set
    validation = spark.read.csv('valid-small-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    # validation = spark.read.csv('valid-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    validation = validation.drop("timestamp")
    validation.createOrReplaceTempView('validation')

    # tuning the hyperparameters by evaluating the model on validation dataset
    users_valid = validation.select("userId").distinct()
    predictions_valid = model.recommendForUserSubset(users_valid, 100)
    predictions_valid = predictions_valid.withColumn("recommendations", explode(col("recommendations"))).select("userId", "recommendations.movieId", "recommendations.rating")
    predictions_valid = predictions_valid.rdd.map(lambda r: ((r.userId, r.movieId), r.rating)).toDF(["user_movie", "predictions_rating"])

    ratingsTuple_valid = validation.rdd.map(lambda r: ((r[0], r[1]), r[2])).toDF(["user_movie", "rating"])
    ratingsTuple_valid.createOrReplaceTempView('ratingsTuple_v')
    predictions_valid.createOrReplaceTempView('predictions_v')
    scoreAndLabels_als_valid = spark.sql('SELECT rating, predictions_rating FROM predictions_v LEFT JOIN ratingsTuple_v ON ratingsTuple_v.user_movie = predictions_v.user_movie').na.drop()

    # Instantiate regression metrics to compare predicted and actual ratings
    metrics_als_valid = RegressionMetrics(scoreAndLabels_als_valid.rdd)

    # Root mean squared error on validation set
    print("Validation set RMSE = %s" % metrics_als_valid.rootMeanSquaredError)

    # R-squared on validation set
    print("Validation set R-squared = %s" % metrics_als_valid.r2)


    # Evaluate the model by computing the RMSE on the test data
    users = test.select("userId").distinct()
    predictions = model.recommendForUserSubset(users, 100)
    predictions = predictions.withColumn("recommendations", explode(col("recommendations"))).select("userId", "recommendations.movieId", "recommendations.rating")
    predictions = predictions.rdd.map(lambda r: ((r.userId, r.movieId), r.rating)).toDF(["user_movie", "predictions_rating"])

    ratingsTuple = test.rdd.map(lambda r: ((r[0], r[1]), r[2])).toDF(["user_movie", "rating"])
    ratingsTuple.createOrReplaceTempView('ratingsTuple')
    predictions.createOrReplaceTempView('predictions')
    scoreAndLabels_als = spark.sql('SELECT rating, predictions_rating FROM predictions LEFT JOIN ratingsTuple ON ratingsTuple.user_movie = predictions.user_movie').na.drop()

    # Instantiate regression metrics to compare predicted and actual ratings
    metrics_als = RegressionMetrics(scoreAndLabels_als.rdd)

    # Root mean squared error
    print("Test set RMSE = %s" % metrics_als.rootMeanSquaredError)

    # R-squared
    print("Test set R-squared = %s" % metrics_als.r2)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('finalproj').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)