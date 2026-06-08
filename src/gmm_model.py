import numpy as np


class GaussianMixtureModel:
   

    def __init__(self, n_components=5, max_iter=100, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.pi = None        # mixture weights
        self.mu = None        # means
        self.sigma = None     # diagonal variances
        self.log_likelihood_history = []


    # Ensure input is 2D
    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    # Initialization
    def _initialize_parameters(self, X):
        X = self._ensure_2d(X)
        n, d = X.shape
        k = self.n_components

        rng = np.random.default_rng(self.random_state)

        # Equal weights
        self.pi = np.ones(k) / k

        # Randomly choose data points as means
        indices = rng.choice(n, k, replace=False)
        self.mu = X[indices]

        # Initialize variances (diagonal)
        var = np.var(X, axis=0) + 1e-6
        self.sigma = np.tile(var, (k, 1))


    # Gaussian PDF
    def _gaussian_pdf(self, X, mu, sigma):
        X = self._ensure_2d(X)

        # clip sigma to avoid extreme values
        diff = X - mu
        exponent = -0.5 * np.sum((diff ** 2) / sigma, axis=1)
        sigma = np.clip(sigma, 1e-6, 1e6)

        # compute log form to avoid overflow
        log_coeff = -0.5 * (X.shape[1] * np.log(2 * np.pi) + np.sum(np.log(sigma)))
        
        return np.exp(log_coeff + exponent)

    # E-step
    def _e_step(self, X):
        X = self._ensure_2d(X)
        n = X.shape[0]
        K = len(self.pi)

        resp = np.zeros((n, K))

        for k in range(K):
            resp[:, k] = self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k])

        # Normalize
        row_sums = resp.sum(axis=1, keepdims=True)

        # avoid division by zero
        row_sums[row_sums == 0] = 1e-10

        resp = resp / row_sums

        return resp

    # M-step
    def _m_step(self, X, responsibilities):
        X = self._ensure_2d(X)

        n, d = X.shape
        K = len(self.pi)

        Nk = responsibilities.sum(axis=0)

        # Update weights
        self.pi = Nk / n
        self.pi = self.pi / np.sum(self.pi)

        # Update means
        self.mu = np.zeros((K, d))
        for k in range(K):
            if Nk[k] < 1e-8:
                # reinitialize this cluster randomly
                self.mu[k] = X[np.random.randint(0, n)]
                self.sigma[k] = np.var(X, axis=0) + 1e-6
                self.pi[k] = 1e-6
                continue

        # Update variances (diagonal)
        self.sigma = np.zeros((K, d))

        for k in range(K):
            diff = X - self.mu[k]
            self.sigma[k] = np.sum(
                responsibilities[:, k][:, None] * (diff ** 2),
                axis=0
            ) / Nk[k]

            # add regularization
            self.sigma[k] += 1e-6

    # Log-likelihood
    def _compute_log_likelihood(self, X):
        X = self._ensure_2d(X)

        n = X.shape[0]
        K = len(self.pi)

        total = np.zeros(n)

        for k in range(K):
            total += self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k])

        total[total == 0] = 1e-10
        return np.sum(np.log(total))

    # Fit model (EM algorithm)
    def fit(self, X):
        X = self._ensure_2d(X)

        self._initialize_parameters(X)

        prev_ll = None

        for _ in range(self.max_iter):

            # E-step
            resp = self._e_step(X)

            # M-step
            self._m_step(X, resp)

            # Log-likelihood
            ll = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(ll)

            # Convergence check
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break

            prev_ll = ll

        return self
    

    def score(self, X):
        return self._compute_log_likelihood(X)
    
    #for pruning
    def prune_components(self, threshold=1e-3):
        mask = self.pi > threshold

        self.pi = self.pi[mask]
        self.mu = self.mu[mask]
        self.sigma = self.sigma[mask]

        # renormalize
        self.pi = self.pi / np.sum(self.pi)