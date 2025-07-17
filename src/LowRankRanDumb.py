import numpy as np
from sklearn.kernel_approximation import RBFSampler

class LowRankRanDumb:
    """
    RanDumb with low-rank residual adapters using Oja's rule.
    - Random Fourier Features embedding (RBFSampler)
    - Online class means and counts
    - Online top-k residual subspace per class via Oja's learning
    """
    def __init__(self, input_dim, embed_dim=2000, gamma='scale', 
                 residual_rank=10, oja_lr=0.1, oja_orthonormalize=True):
        # RFF embedder
        self.embedder = RBFSampler(gamma=gamma, n_components=embed_dim)
        # placeholder until fitting
        self._is_fitted = False
        # class statistics
        self.class_means = {}     # label -> mean vector (embed_dim,)
        self.class_counts = {}    # label -> count int
        # residual subspace per class: W_c of shape (embed_dim, residual_rank)
        self.residual_rank = residual_rank
        self.oja_lr = oja_lr
        self.oja_orth = oja_orthonormalize
        self.class_W = {}         # label -> W matrix

    def fit_embedder(self, X_raw):
        """
        Fit the RBFSampler on raw inputs X_raw of shape (n_samples, input_dim).
        This initializes the random Fourier mapping.
        """
        self.embedder.fit(X_raw)
        self._is_fitted = True

    def transform(self, X_raw):
        if not self._is_fitted:
            self.fit_embedder(X_raw)
        return self.embedder.transform(X_raw)

    def _init_class(self, label, D):
        # initialize mean and count
        self.class_means[label] = np.zeros(D)
        self.class_counts[label] = 0
        # initialize residual subspace
        self.class_W[label] = np.random.randn(D, self.residual_rank) * 0.01
        # orthonormalize if desired
        if self.oja_orth:
            self._orthonormalize(label)

    def _orthonormalize(self, label):
        W = self.class_W[label]
        # QR decomposition
        Q, _ = np.linalg.qr(W)
        self.class_W[label] = Q[:, :self.residual_rank]

    def partial_fit(self, X_raw, y):
        """
        Online update on raw inputs X_raw (n_samples, input_dim) and labels y.
        """
        Z = self.transform(X_raw)
        for z, label in zip(Z, y):
            if label not in self.class_means:
                self._init_class(label, Z.shape[1])

            # update class mean
            n = self.class_counts[label]
            mu = self.class_means[label]
            new_mu = (n * mu + z) / (n + 1)
            self.class_means[label] = new_mu
            self.class_counts[label] = n + 1

            # compute residual
            residual = z - new_mu
            W = self.class_W[label]
            # Oja's learning: update W to capture residual direction
            # for k in rank dimensions
            # incremental: W += lr * (residual[:, None] * (residual @ W)[None, :])
            delta = np.outer(residual, residual @ W)
            W += self.oja_lr * delta
            self.class_W[label] = W
            if self.oja_orth:
                self._orthonormalize(label)

    def predict(self, X_raw):
        """
        Predict labels for raw inputs X_raw.
        Returns array of labels with shape (n_samples,).
        """
        Z = self.transform(X_raw)
        preds = []
        for z in Z:
            best_label = None
            best_dist = np.inf
            for label, mu in self.class_means.items():
                W = self.class_W[label]
                # project residual: c = W.T @ (z - mu)
                c = W.T.dot(z - mu)
                # reconstruct residual: r = W @ c
                r = W.dot(c)
                # compute distance to corrected mean
                dist = np.linalg.norm(z - mu - r)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
            preds.append(best_label)
        return np.array(preds)

    def score(self, X_raw, y_true):
        y_pred = self.predict(X_raw)
        return np.mean(y_pred == y_true)

# Example usage:
# model = LowRankRanDumb(input_dim=32*32*3, embed_dim=2000, residual_rank=10)
# model.partial_fit(train_X_raw, train_y)
# acc = model.score(test_X_raw, test_y)
# print("Accuracy:", acc)
