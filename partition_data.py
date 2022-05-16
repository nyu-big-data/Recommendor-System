import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import countDistinct
from functools import reduce
from pyspark.sql import DataFrame


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

    # list_of_users = ratings.select('userId').distinct().collect()
    # list_of_users = [v['userId'] for v in list_of_users]

    # # print(list_of_users)

    #Counting the number of distinct users 
    count_of_user = ratings.select('userId').distinct().count()

    #Dividing the users in three dataframes 
    temp = ratings.filter((col('userId') <= (2*count_of_user/3)) & (col('userId') >= (count_of_user/3)))
    temp_df = ratings.filter(col('userId') > (2*count_of_user/3))
    temp_df2 = ratings.filter(col('userId') < (count_of_user/3)) 

    #Counting the entries each user has 
    count_user_history = temp_df.select('userId').groupby('userId').count()
    count_user_history.printSchema()
    count_user_history.createOrReplaceTempView('count_user_history')
    count_user_history.show(10)
    
    #Using the window function to partition each user into train and test     
    window_1 = Window.partitionBy(temp_df['userId']).orderBy('timestamp')
    train_1 = temp_df.select('*', F.rank().over(window_1).alias('rank')).filter(F.col('rank') > count_user_history['userId'] * 0.8).drop('rank')
    test = temp_df.select('*', F.rank().over(window_1).alias('rank')).filter(F.col('rank') <= count_user_history['userId'] * 0.8).drop('rank')

    # train_1.show(10)

    count_user_history2 = temp_df2.select('userId').groupby('userId').count()
    #Using the window function to partition each user into train and valid
    window_2 = Window.partitionBy(temp_df2['userId']).orderBy('timestamp')
    train_2 = temp_df2.select('*', F.rank().over(window_2).alias('rank')).filter(F.col('rank') > count_user_history2['userId']* 0.8).drop('rank')
    valid = temp_df2.select('*', F.rank().over(window_2).alias('rank')).filter(F.col('rank') <= count_user_history2['userId'] * 0.8).drop('rank')

    dfs = [train_1, train_2, temp]

    # combining all the train dfs 
    train = reduce(DataFrame.unionAll, dfs)
    train = train.orderBy('userId')
    test = test.orderBy('userId')
    valid = valid.orderBy('userId')

    #Saving to csv 
    train .coalesce(1).write.csv('hdfs:/user/{}/all/trainlarge.csv'.format(netID))
    valid.coalesce(1).write.csv('hdfs:/user/{}/all/validlarge.csv'.format(netID))
    test.coalesce(1).write.csv('hdfs:/user/{}/all/testlarge.csv'.format(netID))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)