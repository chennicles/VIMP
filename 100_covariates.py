import numpy as np
import vimp_loop as vloop
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import sample


#Initialize variables
n = 1000
loops = 50
n_cov = 50
xs = np.random.normal(0,2,n*n_cov)
X = xs.reshape(n, n_cov)
eps = np.random.normal(0,1,n)

#Calculating indicator function terms
I1 = (X[:,0] > -2) & (X[:,0] < 2)
I2 = X[:,1] < 0
I3 = X[:,2] > 0

#y = I1*abs(X[:,0]) + I2 + I3 + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps
y = X[:,0]**2/X[:,1] + X[:,2]**2 + X[:,3] - X[:,4]**3 + X[:,5]**3 + np.cos(X[:,6])/X[:,7] + X[:,8]**3 * abs(X[:,9]/4)**3+ X[:,10]**3 + X[:,11]/10 + (6*X[:,12] - 3*X[:,13])/(X[:,14] - X[:,15]) + np.sin(X[:,16]**3) + 4*X[:,17]**2 + X[:,18]/3- X[:,19] + eps

X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5)

        # Estimating psi1 and psi2
v1 = vloop.vimp(y1,X1)
v2 = vloop.vimp(y2,X2)
psi1 = v1.get(y1,X1)
psi2 = v2.get(y2,X2)
plt.scatter(psi1[0:20], psi2[0:20], color='r')
plt.scatter(psi1[20:], psi2[20:], color='b')
plt.axline((0, 0), slope=1)
plt.show()

sum(1 for x in psi1 if x < 0)
sum(1 for x in psi2 if x < 0)

sum(1 for x in psi1 if x > 1)
sum(1 for x in psi2 if x > 1)

test = vloop.vimp(y, X)
test.get(y, X)
test.mirror()
q = 0.2              #Designate FDR level cutoff q
stats = test.cutoff(q)
stats[0]
stats[1]



n = 100
loops = 1
n_cov = 100
q = 0.3
cutoffs = []
mirrors = []
for i in tqdm(range(loops)):
    xs = np.random.normal(0, 2, n * n_cov)
    X = xs.reshape(n, n_cov)
    eps = np.random.normal(0, 1, n)


    y = X[:,0]**2/X[:,1] + X[:,2]**6 + X[:,3] - X[:,4]**3 + X[:,5]**3 + np.cos(X[:,6])/X[:,7] + X[:,8]**3 * abs(X[:,9]/4)**3
    + X[:,10]**3 + X[:,11]/10 + (10*X[:,12] - 3*X[:,13])/(X[:,14] - X[:,15]) + np.sin(X[:,16]**3) + 15*X[:,17]**2 + X[:,18]/3
    - X[:,19] + eps

    test = vloop.vimp()
    stats = test.datasplit(y, X, q)

    cutoffs.append(stats[0])
    mirrors.append(stats[1])

np.where(sum(mirrors)/50 > np.mean(cutoffs))[0]
