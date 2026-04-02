import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BatchGradientDescent :

    def __init__(self,lr,epochs):
        self.weight = None
        self.bias = None
        self.lr = lr
        self.epochs = epochs

    def fit(self, X_train,y_train):
        self.bias = 0
        self.weight = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            y_hat = np.dot(X_train,self.weight)+self.bias
            der = -2*np.mean(y_train-y_hat)
            der2= np.dot((y_train-y_hat),X_train)/X_train.shape[0]
            self.bias = self.bias - (self.lr*der)
            self.weight = self.weight -(self.lr * der2)





    def predict(self,X_test):
        return np.dot(X_test,self.weight)+self.bias
