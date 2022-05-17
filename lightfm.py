import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score

import time 

data = pd.read_csv('/content/ratings.csv', names= ['userId', 'movieId', 'rating', 'timestamp'])
train = pd.read_csv('/content/train.csv', names= ['userId', 'movieId', 'rating', 'timestamp'])
train = train.drop(['timestamp'], axis = 1)
test = pd.read_csv('/content/test.csv', names= ['userId', 'movieId', 'rating', 'timestamp'])
test = test.drop(['timestamp'], axis = 1)

user_id = data['userId'].unique()[1:].astype('int64')
movie_id = data['movieId'].unique()[1:].astype('int64')

data = Dataset()

data.fit(user_id,  
         movie_id 
        )

train_values = train.values[1:].astype('int64')
test_values = test.values[1:].astype('int64')

train, weights_matrix = data.build_interactions([tuple(i) for i in train_values])
test, weights_matrix = data.build_interactions([tuple(i) for i in test_values])

start_time = time.time()
model = LightFM(loss='warp')
model.fit(train, epochs=50, num_threads=1)
end_time = time.time()

print("Total LightFM model training time: {} seconds".format(end_time - start_time))

training_auc = auc_score(model, train).mean()

print("Training Accuracy : {}".format(training_auc))

# Evaluate the trained model
test_precision = precision_at_k(model, test, k=100).mean()
test_auc = auc_score(model, test).mean()

print("Testing Precision: {}".format(test_precision))
print("Testing Accuracy: {}".format(test_auc))