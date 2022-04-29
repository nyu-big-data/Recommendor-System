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


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    ratings = spark.read.csv('ratings.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER')
    ratings.printSchema()

    ratings.createOrReplaceTempView('ratings')
    # Construct a query
    print('Getting top 100 movies with highest ratings')
    query = spark.sql('SELECT movieID, COUNT(rating) AS counts, (SUM(rating)/COUNT(rating)) AS avg_rating FROM ratings GROUP BY movieID HAVING COUNT(rating) > 0 ORDER BY avg_rating DESC, counts DESC LIMIT 100')

    # # Print the results to the console to delet it 
    query.show()

    # delet dis now
    #####--------------YOUR CODE STARTS HERE--------------#####

    #make sure to load reserves.json, artist_term.csv, and tracks.csv
    #For the CSVs, make sure to specify a schema and delet it!


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('finalproj').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)