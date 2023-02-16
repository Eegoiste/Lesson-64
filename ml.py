import sklearn
sklearn.__version__

import numpy as np
from sklearn import datasets

np.random.seed(10)
raw_data = datasets.load_iris()
#print(raw_data)
#print(raw_data.DESCR)
#print(raw_data.keys())

data = raw_data.data
target = raw_data.target
#print(data.shape)
#print(target.shape)

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
#print(data_train.shape)
#print(target_train.shape)

#print(data_test.shape)
#print(target_test.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data_train, target_train)

model.predict(data_test)
model.fit(data_train, target_train)
