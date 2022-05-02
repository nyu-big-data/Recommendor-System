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


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    ratings = spark.read.csv('ratings.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER')
    # ratings = spark.read.csv('ratings-small.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER')
    ratings.printSchema()

    ratings.createOrReplaceTempView('ratings')
    # Construct a query
    print('Getting top 100 movies with highest ratings')
    predicted_ratings = spark.sql('SELECT movieId, (SUM(rating)/COUNT(rating)) AS predicted_rating FROM ratings GROUP BY movieId HAVING COUNT(rating) > 0 ORDER BY predicted_rating DESC LIMIT 100')

    # Print the predicted ratings to the console
    predicted_ratings.show()
    predicted_ratings.createOrReplaceTempView('predicted_ratings')

    #ratings here will be subbed by the test df
    scoreAndLabels = spark.sql('SELECT rating, predicted_rating FROM predicted_ratings LEFT JOIN ratings ON predicted_ratings.movieId = ratings.movieId')
    scoreAndLabels.show()

    # Instantiate regression metrics to compare predicted and actual ratings
    metrics = RegressionMetrics(scoreAndLabels.rdd)

    # Root mean squared error
    print("RMSE = %s" % metrics.rootMeanSquaredError)

    # R-squared
    print("R-squared = %s" % metrics.r2)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('finalproj').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)