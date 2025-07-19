import numpy as np

class NsCEClassifier:
    """
    Online Nearest‐Class‐Mean with Unknown Prototype (NsCE).
    - Always initializes each class the first time it's seen.
    - For later samples, compares distance to its own class mean:
        • if dist <= threshold → update class mean
        • if dist > threshold → update unknown prototype
    - At predict time, picks nearest class mean if within threshold,
      else returns None.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        # known classes
        self.class_means = {}    # label -> mean vector
        self.class_counts = {}   # label -> count
        # unknown prototype
        self.unknown_mean = None
        self.unknown_count = 0

    def partial_fit(self, Z: np.ndarray, y: np.ndarray):
        """
        Online update for a batch of embeddings Z and labels y.
        Z: (n_samples, D)
        y: (n_samples,)
        """
        for zi, yi in zip(Z, y):
            if yi not in self.class_means:
                # first time we see this class: initialize it
                self.class_means[yi] = zi.copy()
                self.class_counts[yi] = 1
                continue

            # we have a mean for this class already
            mean = self.class_means[yi]
            dist = np.linalg.norm(zi - mean)

            if dist > self.threshold:
                # treat as unknown
                if self.unknown_mean is None:
                    self.unknown_mean = zi.copy()
                    self.unknown_count = 1
                else:
                    n = self.unknown_count
                    self.unknown_mean = (n * self.unknown_mean + zi) / (n + 1)
                    self.unknown_count += 1
            else:
                # update the class mean
                n = self.class_counts[yi]
                self.class_means[yi] = (n * mean + zi) / (n + 1)
                self.class_counts[yi] += 1

    def predict(self, Z: np.ndarray):
        """
        For each embedding in Z, return the nearest known class if within threshold,
        otherwise return None.
        """
        preds = []
        labels = list(self.class_means.keys())
        means = np.stack([self.class_means[c] for c in labels], axis=0)  # (C, D)

        for zi in Z:
            # compute distances to all class means
            dists = np.linalg.norm(means - zi[None, :], axis=1)
            idx = dists.argmin()
            if dists[idx] <= self.threshold:
                preds.append(labels[idx])
            else:
                preds.append(None)
        return np.array(preds, dtype=object)

    def score(self, Z: np.ndarray, y_true: np.ndarray):
        """
        Compute accuracy on (Z, y_true), ignoring any None predictions.
        """
        y_pred = self.predict(Z)
        mask = y_pred != None
        if mask.sum() == 0:
            return 0.0
        return (y_pred[mask] == y_true[mask]).mean()
