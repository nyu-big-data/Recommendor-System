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
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode, col

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    # reading from the partitioned training dataset
    training = spark.read.csv('all/trainsmall/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    # training = spark.read.csv('all/trainlarge/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    
     # reading test data from the partitioned file
    test = spark.read.csv('all/testsmall/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    # test = spark.read.csv('all/testlarge/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()

     # Tune the hyperparameters with the validation data set
    validation = spark.read.csv('all/validsmall/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
    # validation = spark.read.csv('all/validlarge/*.csv', schema='userId INTEGER, movieId INTEGER, rating FLOAT, timestamp INTEGER').na.drop()
   
    training = training.drop("timestamp")
    training.createOrReplaceTempView('training')

    # Construct a query to get the top 100 movies with highest ratings from the training set
    print('Getting top 100 movies with highest ratings')

    start_time = time.time()

    predicted_ratings = spark.sql('SELECT movieId, (SUM(rating)/COUNT(rating)) AS predicted_rating FROM training GROUP BY movieId HAVING COUNT(rating) > 0 ORDER BY predicted_rating DESC LIMIT 100')
    
    end_time = time.time()
    print("Total popularity baseline model query execution time: {} seconds".format(end_time - start_time))

    # Print the predicted ratings to the console
    predicted_ratings.createOrReplaceTempView('predicted_ratings')

   
    test = test.drop("timestamp")
    test.createOrReplaceTempView('test')

    # Getting at most 100 highest rated movie list of each user in the test dataset
    test_ranking = test.withColumn("rn", F.row_number().over(Window.partitionBy("userId").orderBy(F.col("rating").desc()))).filter(f"rn <= {100}").groupBy("userId").agg(F.collect_list(F.col("movieId")).alias("movie_list"))
    test_ranking.createOrReplaceTempView('test_ranking')

    top_100_movies = predicted_ratings.agg(F.collect_list(F.col("movieId")).alias("predicted_movie_list"))

    predicted_ratings_ranking = test_ranking.select("movie_list").crossJoin(top_100_movies.select("predicted_movie_list"))
    

    # joining 2 tables and leaving only the ratings from each table to be compared
    scoreAndLabels = spark.sql('SELECT rating, predicted_rating FROM predicted_ratings INNER JOIN test ON predicted_ratings.movieId = test.movieId').na.drop()

    print("With popularity baseline model: ")

    # Instantiate ranking metrics object
    ranking_metrics = RankingMetrics(predicted_ratings_ranking.rdd)

    print("popularity baseline Mean average precision at 100 = %s" % ranking_metrics.meanAveragePrecisionAt(100))

    print("popularity baseline NDCG at 100 = %s" % ranking_metrics.ndcgAt(100))

    # Instantiate regression metrics to compare predicted and actual ratings
    regression_metrics = RegressionMetrics(scoreAndLabels.rdd)

    # Root mean squared error
    print("Popularity baseline model RMSE = %s" % regression_metrics.rootMeanSquaredError)


    print("---------------------------------------------------")

    print("with ALS model: ")

    # training
    start_time = time.time()

    als = ALS(rank=200, maxIter=15, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True ,coldStartStrategy="drop")
    model = als.fit(training)

    end_time = time.time()
    print("Total ALS model training time: {} seconds".format(end_time - start_time))

    validation = validation.drop("timestamp")
    validation.createOrReplaceTempView('validation')

    # tuning the hyperparameters by evaluating the model on validation dataset
    users_valid = validation.select("userId").distinct()
    predictions_valid = model.recommendForUserSubset(users_valid, 100)
    predictions_valid = predictions_valid.withColumn("recommendations", explode(col("recommendations"))).select("userId", "recommendations.movieId", "recommendations.rating")
    predicted_items_valid = predictions_valid.withColumn("rn", F.row_number().over(Window.partitionBy("userId").orderBy(F.col("rating").desc()))).groupBy("userId").agg(F.collect_list(F.col("movieId")).alias("predicted_movie_list"))
    predictions_valid = predictions_valid.rdd.map(lambda r: ((r.userId, r.movieId), r.rating)).toDF(["user_movie", "predictions_rating"])

    ratingsTuple_valid = validation.rdd.map(lambda r: ((r[0], r[1]), r[2])).toDF(["user_movie", "rating"])
    ratingsTuple_valid.createOrReplaceTempView('ratingsTuple_v')
    predictions_valid.createOrReplaceTempView('predictions_v')
    scoreAndLabels_als_valid = spark.sql('SELECT rating, predictions_rating FROM predictions_v INNER JOIN ratingsTuple_v ON ratingsTuple_v.user_movie = predictions_v.user_movie').na.drop()

    valid_top_user_movies = validation.withColumn("rn", F.row_number().over(Window.partitionBy("userId").orderBy(F.col("rating").desc()))).filter(f"rn <= {100}").groupBy("userId").agg(F.collect_list(F.col("movieId")).alias("movie_list"))

    valid_top_user_movies.createOrReplaceTempView('valid_top_user_movies')
    predicted_items_valid.createOrReplaceTempView('predicted_items_valid')
    rating_metrics_tuple_valid = spark.sql('SELECT movie_list, predicted_movie_list FROM valid_top_user_movies INNER JOIN predicted_items_valid ON predicted_items_valid.userId = valid_top_user_movies.userId').na.drop()

    # Instantiate ranking metrics object
    ranking_metrics_als_valid = RankingMetrics(rating_metrics_tuple_valid.rdd)

    print("ALS validation set Mean average precision at 100 = %s" % ranking_metrics_als_valid.meanAveragePrecisionAt(100))

    print("ALS validation set baseline NDCG at 100 = %s" % ranking_metrics_als_valid.ndcgAt(100))

    
    # Instantiate regression metrics to compare predicted and actual ratings
    metrics_als_valid = RegressionMetrics(scoreAndLabels_als_valid.rdd)

    # Root mean squared error on validation set
    print("ALS Validation set RMSE = %s" % metrics_als_valid.rootMeanSquaredError)


    # Evaluate the model by computing the RMSE on the test data
    users = test.select("userId").distinct()
    predictions = model.recommendForUserSubset(users, 100)
    predictions = predictions.withColumn("recommendations", explode(col("recommendations"))).select("userId", "recommendations.movieId", "recommendations.rating")
    predicted_items_test = predictions.withColumn("rn", F.row_number().over(Window.partitionBy("userId").orderBy(F.col("rating").desc()))).groupBy("userId").agg(F.collect_list(F.col("movieId")).alias("predicted_movie_list"))
    predictions = predictions.rdd.map(lambda r: ((r.userId, r.movieId), r.rating)).toDF(["user_movie", "predictions_rating"])

    ratingsTuple = test.rdd.map(lambda r: ((r[0], r[1]), r[2])).toDF(["user_movie", "rating"])
    ratingsTuple.createOrReplaceTempView('ratingsTuple')
    predictions.createOrReplaceTempView('predictions')
    scoreAndLabels_als = spark.sql('SELECT rating, predictions_rating FROM predictions INNER JOIN ratingsTuple ON ratingsTuple.user_movie = predictions.user_movie').na.drop()

    predicted_items_test.createOrReplaceTempView('predicted_items_test')
    rating_metrics_tuple_test = spark.sql('SELECT movie_list, predicted_movie_list FROM test_ranking INNER JOIN predicted_items_test ON predicted_items_test.userId = test_ranking.userId').na.drop()
    # Instantiate ranking metrics object
    ranking_metrics_als_test = RankingMetrics(rating_metrics_tuple_test.rdd)

    print("ALS test set Mean average precision at 100 = %s" % ranking_metrics_als_test.meanAveragePrecisionAt(100))

    print("ALS test set baseline NDCG at 100 = %s" % ranking_metrics_als_test.ndcgAt(100))

    # Instantiate regression metrics to compare predicted and actual ratings
    metrics_als = RegressionMetrics(scoreAndLabels_als.rdd)

    # Root mean squared error
    print("Test set RMSE = %s" % metrics_als.rootMeanSquaredError)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('finalproj').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)