import numpy as np
import vimp_loop as vloop
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import sample


#Initialize variables
n = 1000
n_cov = 15
xs = np.random.normal(0,2,n*n_cov)
X = xs.reshape(n, n_cov)
eps = np.random.normal(0,1,n)

#Calculating indicator function terms
I1 = (X[:,0] > -2) & (X[:,0] < 2)
I2 = X[:,1] < 0
I3 = X[:,2] > 0

#y = I1*abs(X[:,0]) + I2 + I3 + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps
y = y = I1*abs(X[:,0]) + I2 + I3 + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5)

        # Estimating psi1 and psi2
v1 = vloop.vimp(y1,X1)
v2 = vloop.vimp(y2,X2)
psi1 = v1.get(y1,X1)
psi2 = v2.get(y2,X2)

sig = [0,1,2,5,6,10]
notsig = [3,4,7,8,9,11,12,13,14]
plt.scatter([psi1[i] for i in sig], [psi2[i] for i in sig], color='r')
plt.scatter([psi1[i] for i in notsig], [psi2[i] for i in notsig], color='b')
plt.axline((0, 0), slope=1)
plt.show()

sum(1 for x in psi1 if x < 0)
sum(1 for x in psi2 if x < 0)

sum(1 for x in psi1 if x > 1)
sum(1 for x in psi2 if x > 1)
