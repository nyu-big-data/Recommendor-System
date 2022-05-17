import pandas as pd
import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

import time 

train = pd.read_csv('/content/trainsmall.csv', names= ['userId', 'movieId', 'rating', 'timestamp'])
users = train["userId"].values
products = train["movieId"].values
ratings = train["rating"].values
train = sp.coo_matrix((ratings, (users, products)))
# train.head()
# Instantiate and train the model

start_time = time.time()
model = LightFM(loss='warp')
model.fit(train, epochs=50, num_threads=1)
end_time = time.time()

print("Total LightFM model training time: {} seconds".format(end_time - start_time))

# test = pd.read_csv('/content/testsmall.csv', names= ['userId', 'movieId', 'rating', 'timestamp'])
# users = test["userId"].values
# products = test["movieId"].values
# ratings = test["rating"].values
# test = sp.coo_matrix((ratings, (users, products)))

# Evaluate the trained model
# test_precision = precision_at_k(model, test, k=5).mean()