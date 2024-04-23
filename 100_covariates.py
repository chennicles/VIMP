import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import vimp_loop as vloop

#Initialize variables
n = 1000
n_cov = 100
xs = np.random.normal(0,2,n*n_cov)
X = xs.reshape(n, n_cov)
eps = np.random.normal(0,1,n)

#Calculating indicator function terms
I1 = (X[:,0] > -2) & (X[:,0] < 2)
I2 = X[:,1] < 0
I3 = X[:,2] > 0

y = I1*abs(X[:,0]) + I2 + I3 + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps


test = vloop.vimp(y, X)
test.get(5)
q = 0.3              #Designate FDR level cutoff q
stats = test.datasplit(q)
stats[0]
stats[1]
