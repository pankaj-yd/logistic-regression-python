import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc
import model

input = input("ENTER YOUR FEATURES: ").split()
dataset = []
for data in input:
    dataset.append(float(data))
dataset = np.asarray(dataset)
dataset = dataset[np.newaxis, :]
dataset = np.c_[np.ones((dataset.shape[0], 1)), dataset]


def load_data(path, header):
    data_df = pd.read_csv(path, header=header)
    return data_df

if __name__ == "__main__":
    data = load_data('./data.txt', None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    #print(X.shape[1])
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    #print(X.shape)
    #print('-'*30)
    #print(y)
    #print(theta.shape)
    #print(y.flatten())

    #print(X)
    #print(probability(theta, X))
    model = model.model()
    model.fit(X, y, theta)
    parameters = model.w_
    print("-"*60)
    print("The model parameters using Gradient descent are")
    print(parameters)
    print("-"*60)

    if model.predict(dataset) < 0.5:
        print('NO')
    else:
        print('YES')
