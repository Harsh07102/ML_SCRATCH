import numpy as np
import matplotlib.pyplot as plt 

class GradientDescent:

    def __init__(self,epochs,lr):
        self.epochs =epochs
        self.lr = lr
        self.m = 100
        self.b = -120

    def fit(self , X,y):
        for i in range(self.epochs):
            slope_b = -2*(np.sum(y - self.m*X.ravel()-self.b))
            slope_m = -2* np.sum((y-self.m*X.ravel()-self.b)*X.ravel())
            self.b = self.b - (self.lr * slope_b)
            self.m = self.m - (self.lr * slope_m)
        print(self.b)
        print(self.m)

    def predict(self,X):
        return self.m*X + self.b