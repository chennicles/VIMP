import numpy as np
import vimp_loop as vloop
from tqdm import tqdm


#Initialize variables
n = 500
loops = 50
n_cov = 100
xs = np.random.normal(0,2,n*n_cov)
X = xs.reshape(n, n_cov)
eps = np.random.normal(0,1,n)

#Calculating indicator function terms
I1 = (X[:,0] > -2) & (X[:,0] < 2)
I2 = X[:,1] < 0
I3 = X[:,2] > 0

y = I1*abs(X[:,0]) + I2 + I3 + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps

test = vloop.vimp()
test.get(y, X)

q = 0.3              #Designate FDR level cutoff q
stats = test.datasplit(y,X,q)
stats[0]
stats[1]



n = 100
loops = 50
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

np.where(sum(mirrors)/50 > n
