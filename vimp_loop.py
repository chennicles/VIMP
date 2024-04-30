import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV


class vimp:
    def get(self, y, x):
        self.y = y
        self.x = x
        # Transform the entire predictor matrix
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(self.x)

        # Fit the full model using ElasticNetCV
        full_model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], max_iter=2000, tol=1e-2)
        full_model.fit(X_poly, self.y)
        u = full_model.predict(X_poly)

        # Compute the full model residual
        full_residual = np.mean((self.y - u) ** 2)

        # Array to store psi values for each predictor
        psi_values = []

        # Loop over each predictor
        for i in range(self.x.shape[1]):
            # Create polynomial features for the current predictor
            X_single = np.delete(self.x, i, axis=1)
            X_single_poly = PolynomialFeatures(degree=2).fit_transform(X_single)

            # Fit the model for the current predictor
            single_model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], max_iter=2000, tol=1e-2)
            single_model.fit(X_single_poly, u)
            u_single = single_model.predict(X_single_poly)

            # Calculate the residual for the model with the current predictor
            residual_single = np.mean((self.y - u_single) ** 2)

            # Calculate psi for the current predictor
            psi = 1 - (full_residual / np.var(self.y)) - (1 - (residual_single / np.var(self.y)))
            psi_values.append(psi)

        return psi_values


    def datasplit(self, y, X, q):
        self.y = y
        self.x = X
        n = len(self.x)
        y1 = self.y[:int(n / 2)]
        y2 = self.y[int(n / 2):]

        X1 = self.x[:int(n / 2)]
        X2 = self.x[int(n / 2):]

        # Estimating psi1 and psi2

        v1 = self.get(y1, X1)
        v2 = self.get(y2, X2)

        psi1 = np.array(v1)
        psi2 = np.array(v2)

        # Calculate mirror stat
        m = abs(psi1 + psi2) - abs(psi1 - psi2)

        # Designate FDR level
        # Assuming M contains p-values from multiple hypothesis tests
        self.q = q  # Your designated FDR level

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

        return tau_q, m



