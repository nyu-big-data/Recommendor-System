#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit recommender.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

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

    ratings = spark.read.csv('ratings.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER').na.drop()
    # ratings = spark.read.csv('ratings-small.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER').na.drop()
    ratings.printSchema()
    ratings.show()
    ratings.createOrReplaceTempView('ratings')

    # Construct a query
    print('Getting top 100 movies with highest ratings')
    predicted_ratings = spark.sql('SELECT movieId, (SUM(rating)/COUNT(rating)) AS predicted_rating FROM ratings GROUP BY movieId HAVING COUNT(rating) > 0 ORDER BY predicted_rating DESC LIMIT 100')

    # Print the predicted ratings to the console
    predicted_ratings.show()
    predicted_ratings.createOrReplaceTempView('predicted_ratings')

    # test = spark.read.csv('test-small-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER').na.drop()
    test = spark.read.csv('test-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER').na.drop()
    
    test.createOrReplaceTempView('test')

    # joining 2 tables and leaving only the ratings from each table to be compared
    scoreAndLabels = spark.sql('SELECT rating, predicted_rating FROM predicted_ratings LEFT JOIN test ON predicted_ratings.movieId = test.movieId').na.drop()
    scoreAndLabels.show()

    # Instantiate regression metrics to compare predicted and actual ratings
    metrics = RegressionMetrics(scoreAndLabels.rdd)

    print("With popularity baseline model: ")

    # Root mean squared error
    print("RMSE = %s" % metrics.rootMeanSquaredError)

    # R-squared
    print("R-squared = %s" % metrics.r2)

    print("---------------------------------------------------")

    print("with ALS model: ")

    # we will sub the training and test dataset with the one we partitioned
    # training = spark.read.csv('train-small-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER').na.drop()
    training = spark.read.csv('train-2.csv/*.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER').na.drop()
    
    training.createOrReplaceTempView('training')
    training.show()
    test.show()

    # training
    als = ALS(maxIter=20, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    users = test.select(als.getUserCol()).distinct()
    predictions = model.recommendForUserSubset(users, 100)
    predictions = predictions.withColumn("recommendations", explode(col("recommendations"))).select("userId", "recommendations.movieId", "recommendations.rating")
    predictions.printSchema()
    predictions = predictions.rdd.map(lambda r: ((r.userId, r.movieId), r.rating)).toDF(["user_movie", "predictions_rating"])
    predictions.show()

    ratingsTuple = test.rdd.map(lambda r: ((r[0], r[1]), r[2])).toDF(["user_movie", "rating"])
    ratingsTuple.show()
    ratingsTuple.createOrReplaceTempView('ratingsTuple')
    predictions.createOrReplaceTempView('predictions')
    scoreAndLabels_als = spark.sql('SELECT rating, predictions_rating FROM predictions LEFT JOIN ratingsTuple ON ratingsTuple.user_movie = predictions.user_movie').na.drop()

    # Instantiate regression metrics to compare predicted and actual ratings
    metrics_als = RegressionMetrics(scoreAndLabels_als.rdd)

    # Root mean squared error
    print("RMSE = %s" % metrics_als.rootMeanSquaredError)

    # R-squared
    print("R-squared = %s" % metrics_als.r2)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('finalproj').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)