import numpy as np
from sklearn.linear_model import ElasticNetCV


class vimp:
    def get(self, y, x):
        self.x = x
        self.y = y
        # Transform the entire predictor matrix

        # Fit the full model using ElasticNetCV
        full_model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
        full_model.fit(self.x, self.y)
        u = full_model.predict(self.x)

        psi = []

        # Loop over each predictor
        for i in range(self.x.shape[1]):
            X_s = np.delete(self.x, i, axis=1)

            # Fit the model for the current predictor
            single_model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
            single_model.fit(X_s, u)
            u_s = single_model.predict(X_s)


            # Calculate psi for the current predictor
            psi1 = 1 - (np.mean((self.y - u) ** 2) / np.var(self.y)) - (1 - np.mean((self.y - u_s) ** 2) / np.var(self.y))
            psi.append(psi1)

        return psi

    def datasplit(self, q):
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


