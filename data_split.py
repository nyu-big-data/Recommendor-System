import pandas as pd 
from sklearn.model_selection import train_test_split


df = pd.read_csv("C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/ratings.csv")

temp_df = df.iloc[:int(len(df)/2)]
temp_df2 = df.iloc[int(len(df)/2):]

train_1, test = train_test_split(temp_df, test_size=0.2)
train_2, valid = train_test_split(temp_df2, test_size=0.2)

# print(train_2.head())

train = train_1.append(train_2, ignore_index=True)
# print(train.head())

train.to_csv('C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/train-small.csv')
test.to_csv('C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/test-small.csv')
valid.to_csv('C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/val-small.csv')
