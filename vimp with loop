import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class vimp:
    def __init__(self, y, x, d):
        self.y = y
        self.x = x
        self.d = d

    def get(self):
        poly = PolynomialFeatures(degree=self.d)
        X_poly = poly.fit_transform(self.x)

        model = LinearRegression()
        model.fit(X_poly, self.y)
        u = model.predict(X_poly)

        nvar = self.x.shape[1]
        psi = [0] * nvar
        mu = [0] * nvar

        for i in range(nvar):
            # Create a new subset of x by deleting one column
            X_sub = np.delete(self.x, i, axis=1)
            # Apply PolynomialFeatures to the new subset
            poly_sub = PolynomialFeatures(degree=self.d)
            X_sub_poly = poly_sub.fit_transform(X_sub)

            # Fit a new model on the transformed subset
            model_sub = LinearRegression()
            model_sub.fit(X_sub_poly, self.y)
            u_sub = model_sub.predict(X_sub_poly)

            # Calculate mu using the coefficients from the new model
            mu[i] = u_sub

            # Calculate variance explained for each model
            residual_var = np.mean((self.y - u) ** 2)
            mu_var = np.mean((self.y - mu[i]) ** 2)
            psi[i] = 1 - (residual_var / np.var(self.y)) - (1 - (mu_var / np.var(self.y)))

        return psi
