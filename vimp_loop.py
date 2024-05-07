import numpy as np
from sklearn.tree import DecisionTreeRegressor  # For regression trees
from sklearn.model_selection import train_test_split




class vimp:
    def __init__(self, y, x):
        self.y = y
        self.x = x
    def get(self, y, x):
        self.y = y
        self.x = x

        # Transform the entire predictor matrix
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        tree_model = DecisionTreeRegressor()
        tree_model.fit(X_train, y_train)

        u = tree_model.predict(self.x)

        # Compute the full model residual
        full_residual = np.mean((self.y - u) ** 2)

        # Array to store psi values for each predictor
        psi_values = []

        # Loop over each predictor
        for i in range(self.x.shape[1]):
            # Create polynomial features for the current predictor
            X_s = np.delete(self.x, i, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X_s, u, test_size=0.2)
            tree_model = DecisionTreeRegressor()
            tree_model.fit(X_train, y_train)

            u_s = tree_model.predict(X_s)

            # Calculate the residual for the model with the current predictor
            residual_single = np.mean((self.y - u_s) ** 2)

            # Calculate psi for the current predictor
            psi = 1 - (full_residual / np.var(self.y)) - (1 - (residual_single / np.var(self.y)))
            psi_values.append(psi)

        return psi_values


    def mirror(self):
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
        return m


    def cutoff(self, q):
        m = self.mirror()
        # Designate FDR level
        # Assuming M contains p-values from multiple hypothesis tests
        self.q = q  # Your designated FDR level

        # Sort the p-values in ascending order
        sorted_m = np.sort(m)
        num_null_rejected = 0

        tau_q = 0
        for i, t in enumerate(sorted_m):
            # Calculate the critical value for this p-value
            fdp = (num_null_rejected + 1) / (i + 1)

            # Find the largest p-value where the p-value is less than the critical value
            if fdp <= q:
                tau_q = t
            if t <= q:
                num_null_rejected += 1

        return tau_q, m


