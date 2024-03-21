import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#Setting 1
n = 5000
x1 = np.random.uniform(-1,1,n)
x2 = np.random.uniform(-1,1,n)
X = np.column_stack((x1, x2))
eps = np.random.normal(0,1,n)
y = X[:,-0]**2 * (X[:,-0] + (7/5)) + (25/9)*X[:,-1]**2 + eps

test = vimp(y,X)
test.get()
#Psi1 = 0.158
#Psi2 = 0.342



#Setting 2
n = 5000
x1 = np.random.uniform(-1,1,n)
x2 = np.random.uniform(-1,1,n)
X = np.column_stack((x1, x2))
eps = np.random.normal(0,1,n)
y = (25/9)*X[:,-1]**2 + eps

test = vimp(y,X)
test.get()
#Psi1 = 0.407
#Psi2 = 0


