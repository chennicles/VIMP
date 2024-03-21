import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class vimp:
  def __init__(self, y, x):
    self.y = y
    self.x = x
    
  def get2(self):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
  
  
    #Choosing technique to estimate conditional mean
    #Regress Y on X using same technique
    model = LinearRegression()
    model.fit(X_poly, y)
    u = model.predict(X_poly)
    
    
    #Regress u on X_-s using same technique
    coefficients = np.polyfit(X[:,1], u, 2)
    u1 = np.polyval(coefficients, X[:,1])
    
    coefficients = np.polyfit(X[:,0], u, 2)
    u2 = np.polyval(coefficients, X[:,0])
    
    psi1 = 1 - (np.mean((y-u)**2)/np.var(y)) - (1 - np.mean((y - u1)**2)/np.var(y))
    psi2 = 1 - (np.mean((y-u)**2)/np.var(y)) - (1 - np.mean((y - u2)**2)/np.var(y))
    
    return f"psi1 =",psi1,"and psi2 =",psi2
  

