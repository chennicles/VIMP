import numpy as np
import loess as ls
import sklearn as sk
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split



#Initialize variables
n = 1000
xs = np.random.normal(0,2,n*15)
X = xs.reshape(n, 15)
eps = np.random.normal(0,1,n)

#Calculating indicator function terms
I1 = (X[:,0] > -2) & (X[:,0] < 2)
I2 = X[:,1] < 0
I3 = X[:,2] > 0


y = I1*X[:,0]*abs(X[:,0]) + I2*X[:,1] + I3*X[:,2] + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps


#Get conditional mean
grad = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42)
model = grad.fit(X, y)
u = model.predict(X)
model1 = grad.fit(X[:,5:15], y)
u1 = model1.predict(X[:,5:15])
model2 = grad.fit(X[:,[0,1,2,3,4,10,11,12,13,14]], y)
u2 = model2.predict(X[:,[0,1,2,3,4,10,11,12,13,14]])
model3 = grad.fit(X[:,0:10], y)
u3 = model3.predict(X[:,0:10])


#Calculate Psi from equation
psi1 = 1 - (np.mean((y-u)**2)/np.var(y)) - (1 - np.mean((y - u1)**2)/np.var(y))
psi2 = 1 - (np.mean((y-u)**2)/np.var(y)) - (1 - np.mean((y - u2)**2)/np.var(y))
psi3 = 1 - (np.mean((y-u)**2)/np.var(y)) - (1 - np.mean((y - u3)**2)/np.var(y))
