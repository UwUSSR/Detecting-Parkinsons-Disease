#importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



#data collection and preprocessing
# loading the data:
df = pd.read_csv('parkinsons.csv')
df.head()

# shape of the data:
df.shape

# 5-pointer summary:
df.describe()

# basic info about data:
df.info()
df ['status'].value_counts()

# separating the dependent and independent features:
x = df.drop(columns=['status', 'name'])
y=df['status']

# standardize the data:
scaler = StandardScaler()
x_std = scaler.fit_transform(x)


#train test split
x_train,x_test, y_train, y_test = train_test_split(x_std, y, test_size = 0.2, stratify=y, random_state=2)
x_train.shape, x_test.shape

# model training:
model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)


#model evaluation
# accuracy score on the training data:
pred_train = model.predict(x_train)
accuracy_score(pred_train, y_train)

# accuracy score on the test data:
pred_test = model.predict(x_test)
accuracy_score(pred_test, y_test)


#final prediction
input_data = (223.365, 238.987, 98.664, 0.00001, 0.00154, 0.00151, 0.00461,
              0.01906, 0.165, 0.01013, 0.01296, 0.0134, 0.03156, 0.098555,
              0, 0, 0, 0, 0, 0, 0, 0)

# convert to numpy array and standardize:
input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshape)

# make the prediction
prediction = model.predict(std_data)
print(prediction)
if prediction == 0:
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person does have Parkinson's Disease")