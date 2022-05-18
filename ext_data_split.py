import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import countDistinct
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import percent_rank


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    print('Reading the data')
    ratings = spark.read.csv('ratings.csv', schema='userId INTEGER, movieId INTEGER, rating DOUBLE, timestamp INTEGER')
    ratings.printSchema()

    #Using the window function to partition each user into train and test     
    window_1 = Window.partitionBy(ratings['userId']).orderBy('timestamp')
    train = ratings.select('*', percent_rank().over(window_1).alias('rank')).filter(F.col('rank') <= .8).drop('rank')
    test = ratings.select('*', percent_rank().over(window_1).alias('rank')).filter(F.col('rank') > .8).drop('rank')

    train = train.orderBy('userId')
    test = test.orderBy('userId')

    #Saving to csv 
    train .coalesce(1).write.csv('hdfs:/user/{}/ext/train'.format(netID))
    test.coalesce(1).write.csv('hdfs:/user/{}/ext/test'.format(netID))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)