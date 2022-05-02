import pandas as pd 
from sklearn.model_selection import train_test_split


df = pd.read_csv("C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/ratings.csv")

temp_df = df.iloc[:int(len(df)/2)].reset_index(drop=True)
temp_df2 = df.iloc[int(len(df)/2):].reset_index(drop=True)

temp_df.sort('timestamp')
temp_df2.sort('timestamp')

# train_1, test = train_test_split(temp_df, test_size=0.2)
# train_2, valid = train_test_split(temp_df2, test_size=0.2)

# print(train_2.head())
n = int(3*len(temp_df)/4)
train_1 = temp_df.iloc[:n]
valid = temp_df.iloc[n:]

train_2 = temp_df2.iloc[:n]
test = temp_df2.iloc[n:]

train = train_1.append(train_2, ignore_index=True).reset_index(drop=True)
# print(train.head())

train.set_index('userId').to_csv('C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/train.csv')
test.set_index('userId').to_csv('C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/test.csv')
valid.set_index('userId').to_csv('C:/Users/Garima Gupta/Desktop/NYU/Big-Data/ml-latest-small/val.csv')
