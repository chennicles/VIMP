import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#Initialize variables
n = 2000
xs = np.random.normal(0,2,n*15)
X = xs.reshape(n, 15)
eps = np.random.normal(0,1,n)

#Calculating indicator function terms
I1 = (X[:,0] > -2) & (X[:,0] < 2)
I2 = X[:,1] < 0
I3 = X[:,2] > 0

#Splitting data
y = I1*abs(X[:,0]) + I2 + I3 + abs(X[:,5]/4)**3 + abs(X[:,6]/4)**5 + (7/3)*np.cos(X[:,10]/2) + eps
y1 = y[:int(n/2)]
y2 = y[int(n/2):]

X1 = X[:int(n/2)]
X2 = X[int(n/2):]


#Estimating psi1 and psi2
v1 = vimp(y1, X1, 3)
v2 = vimp(y2, X2, 3)

psi1 = np.array(v1.get())
psi2 = np.array(v2.get())


#Calculate mirror stat
m = abs(psi1 + psi2) - abs(psi1 - psi2)


#Designate FDR level
# Assuming M contains p-values from multiple hypothesis tests
q = 0.1  # Your designated FDR level

# Sort the p-values in ascending order
sorted_m = np.sort(m)
n = len(sorted_m)

# Calculate the cutoff tau_q using BH procedure
tau_q = 0
for i, t in enumerate(sorted_m):
    # Calculate the critical value for this p-value
    critical_value = (i + 1) / n * q

    # Find the largest p-value where the p-value is less than the critical value
    if t <= critical_value:
        tau_q = t

print("tau_q:", tau_q)

