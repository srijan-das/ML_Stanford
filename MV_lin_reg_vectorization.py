import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/srijan-das/mlaicrc/master/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')

features = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']

X = data[features]
y = np.array(data['Price'])

for f in features:
    X[f] = (X[f]) / (X[f].max())
'''
for f in features:
    print("Linear Corr coeff b/w {} and price {}".format(f, np.corrcoef(X[f].values, y)[0,1]))

features = [f for f in features if np.corrcoef(X[f].values, y)[0,1] > 0.4]

X = X[features]
'''

X['C'] = np.zeros(len(X)) + 1
X = X[['C', 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

X = X.values

theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y)

print("Prediction : {} \nActual : {}".format(np.dot(X[4],theta),y[4]))
print("Accuracy = ",(np.dot(X[4],theta) - y[4]) / y.mean())