import numpy as np


class GaussianHMM:
    def __init__(self, n_states=3):
        self.n_states = n_states

        self.pi = None       # initial probabilities (S,)
        self.A = None        # transition matrix (S,S)
        self.mu = None       # means (S, D)
        self.sigma = None    # variances (S, D)


    def initialize(self, sequences):
        """
        sequences: list of (T, D) arrays
        """
        S = self.n_states
        D = sequences[0].shape[1]

        # Initial state probability
        self.pi = np.zeros(S)
        self.pi[0] = 1.0   # always start from state 0

        # Left-to-right transition (important!)
        self.A = np.zeros((S, S))
        for i in range(S):
            if i < S - 1:
                self.A[i, i] = 0.6
                self.A[i, i+1] = 0.4
            else:
                self.A[i, i] = 1.0

        # Initialize Gaussian parameters
        all_data = np.vstack(sequences)

        self.mu = np.zeros((S, D))
        self.sigma = np.zeros((S, D))

        # split data into S chunks
        chunks = np.array_split(all_data, S)

        for s in range(S):
            self.mu[s] = np.mean(chunks[s], axis=0)
            self.sigma[s] = np.var(chunks[s], axis=0) + 1e-6

    
    def _gaussian_prob(self, x, mu, sigma):
        sigma = np.clip(sigma, 1e-6, 1e6)

        diff = x - mu
        exponent = -0.5 * np.sum((diff ** 2) / sigma)

        log_coeff = -0.5 * (
            len(x) * np.log(2 * np.pi) + np.sum(np.log(sigma))
        )

        return np.exp(log_coeff + exponent)
    

    def forward(self, sequence):
        T = sequence.shape[0]
        S = self.n_states

        alpha = np.zeros((T, S))

        # Initialization
        for s in range(S):
            alpha[0, s] = self.pi[s] * self._gaussian_prob(sequence[0], self.mu[s], self.sigma[s])

        # Recursion
        for t in range(1, T):
            for s in range(S):
                prob = 0
                for prev in range(S):
                    prob += alpha[t-1, prev] * self.A[prev, s]

                alpha[t, s] = prob * self._gaussian_prob(sequence[t], self.mu[s], self.sigma[s])


        total = np.sum(alpha[-1])

        if total<=0:
            total = 1e-300

        return np.log(total)
    
    def backward(self, sequence, scale):
        T = sequence.shape[0]
        S = self.n_states

        beta = np.zeros((T, S))

        beta[T-1] = np.ones(S) / scale[T-1]

        for t in range(T-2, -1, -1):
            for s in range(S):
                total = 0
                for j in range(S):
                    total += self.A[s, j] * self._gaussian_prob(sequence[t+1], self.mu[j], self.sigma[j]) * beta[t+1, j]

                beta[t, s] = total / scale[t]

        return beta
        
    
    def compute_gamma(self, alpha, beta):
        gamma = alpha * beta

        # normalize
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-10)

        return gamma

    def compute_xi(self, sequence, alpha, beta):
        T = sequence.shape[0]
        S = self.n_states

        xi = np.zeros((T-1, S, S))

        for t in range(T-1):
            denom = 0

            for i in range(S):
                for j in range(S):
                    val = alpha[t, i] * self.A[i, j] * \
                        self._gaussian_prob(sequence[t+1], self.mu[j], self.sigma[j]) * \
                        beta[t+1, j]

                    xi[t, i, j] = val
                    denom += val

            xi[t] /= (denom + 1e-10)

        return xi
    

    def baum_welch(self, sequences, n_iter=10):
        S = self.n_states
        D = sequences[0].shape[1]

        for _ in range(n_iter):

            pi_new = np.zeros(S)
            A_new = np.zeros((S, S))
            mu_new = np.zeros((S, D))
            sigma_new = np.zeros((S, D))

            gamma_sum = np.zeros(S)

            for seq in sequences:
                alpha, scale = self._forward_matrix(seq)
                beta = self.backward(seq, scale)

                gamma = self.compute_gamma(alpha, beta)
                xi = self.compute_xi(seq, alpha, beta)

                pi_new += gamma[0]

                A_new += np.sum(xi, axis=0)

                for s in range(S):
                    gamma_s = gamma[:, s][:, None]
                    mu_new[s] += np.sum(gamma_s * seq, axis=0)
                    sigma_new[s] += np.sum(gamma_s * (seq - self.mu[s])**2, axis=0)
                    gamma_sum[s] += np.sum(gamma[:, s])

            # Normalize
            pi_sum = np.sum(pi_new)

            if pi_sum == 0:
                pi_sum = 1e-10

            self.pi = pi_new / pi_sum

            A_new += 1e-6
            row_sums = np.sum(A_new, axis=1, keepdims=True)
            self.A = A_new / row_sums

            for s in range(S):
                if gamma_sum[s] < 1e-10:
                    gamma_sum[s] = 1e-10

                self.mu[s] = mu_new[s] / gamma_sum[s]
                self.sigma[s] = sigma_new[s] / gamma_sum[s] + 1e-6

    def _forward_matrix(self, sequence):
        T = sequence.shape[0]
        S = self.n_states

        alpha = np.zeros((T, S))
        scale = np.zeros(T)

        # init
        for s in range(S):
            alpha[0, s] = self.pi[s] * self._gaussian_prob(sequence[0], self.mu[s], self.sigma[s])

        scale[0] = np.sum(alpha[0])
        if scale[0] == 0:
            scale[0] = 1e-10
        alpha[0] /= scale[0]

        # recursion
        for t in range(1, T):
            for s in range(S):
                alpha[t, s] = sum(
                    alpha[t-1, prev] * self.A[prev, s]
                    for prev in range(S)
                ) * self._gaussian_prob(sequence[t], self.mu[s], self.sigma[s])

            scale[t] = np.sum(alpha[t])
            if scale[t] == 0:
                scale[t] = 1e-10
            alpha[t] /= scale[t]

        return alpha, scale