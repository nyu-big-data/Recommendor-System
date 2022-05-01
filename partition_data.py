import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.sql import Window
from pyspark.sql.functions import col


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

    c = int(ratings.count() // 2)
    temp_df = ratings.sort('userId').limit(c)
    temp_df2 = ratings.sort('userId', ascending=False).limit(c)

    # temp_df = temp_df.groupby('userId').orderby('timestamp')
    # temp_df2 = temp_df2.groupby('userId').orderby('timestamp')
    n = 8

    window_1 = Window.partitionBy(temp_df['userId']).orderBy('timestamp')
    train_1 = temp_df.select('*', F.rank().over(window_1).alias('rank')).filter(F.col('rank') > n).drop('rank')
    test = temp_df.select('*', F.rank().over(window_1).alias('rank')).filter(F.col('rank') <= n).drop('rank')

    window_2 = Window.partitionBy(temp_df2['userId']).orderBy('timestamp')
    train_2 = temp_df2.select('*', F.rank().over(window_2).alias('rank')).filter(F.col('rank') <= n).drop('rank')
    valid = temp_df2.select('*', F.rank().over(window_2).alias('rank')).filter(F.col('rank') > n).drop('rank')

    train = train_1.union(train_2)

    #Saving to csv 
    train .coalesce(1).write.csv('hdfs:/user/{}/train-small.csv'.format(netID))
    valid.coalesce(1).write.csv('hdfs:/user/{}/valid-small.csv'.format(netID))
    test.coalesce(1).write.csv('hdfs:/user/{}/test-small.csv'.format(netID))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)