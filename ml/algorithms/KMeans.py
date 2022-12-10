import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k, max_iter, n_init) -> None:
        self.clusters = k
        self.old_centres = None
        self.max_iter = max_iter
        self.n_init = n_init
        self.results = []


    def fit(self, X):
        for i in range(self.n_init):
            self.centres = self._initialise_centroids(X)

            for epoch in range(self.max_iter):
                distances = self._compute_distance(X, self.centres)
                labels = np.argmin(distances, axis=0)
                self.old_centres = self.centres
                self.centres = self._update_centres(X, labels)
                #self._plot_clustering(X, labels)
                if (self.centres == self.old_centres).all():
                    break

            distance = np.sum(self._compute_distance(X, self.centres), axis=1)
            self.results.append(np.array([self.centres, labels, np.sum(distance)]))

        self.results = np.array(self.results)
        min_distance = np.min(self.results[:, 2], axis=0)
        return (self.results[self.results[:, 2] == min_distance])[0]


    def _compute_distance(self, X, centres):
        return np.array([np.sum((X - centre)**2, axis=1) for centre in centres])


    def _initialise_centroids(self, X):
        return X[np.random.choice(X.shape[0], self.clusters), :]

    def _update_centres(self, X, labels):
        return np.array([X[labels == cluster].mean(axis=0) for cluster in range(self.clusters)])
    
    def _plot_clustering(self, X, labels):
        plt.scatter(X[:, 0], X[:, 1], s=7, c=labels)
        plt.scatter(self.centres[:,0], self.centres[:,1], marker='o', c='r', s=100, label='Cluster centers')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    iris = datasets.load_iris()
    iris_data = pd.DataFrame(data=iris["data"], columns=iris.feature_names)
    X = (iris_data.iloc[:, [0, 2]]).to_numpy()
    kmeans = KMeans(k=2, max_iter=10, n_init=3)
    results = kmeans.fit(X)
    plt.scatter(X[:, 0], X[:, 1], s=7, c=results[1])
    plt.scatter(results[0][0], results[0][1], marker='o', c='r', s=100, label='Best Cluster centers')
    plt.legend()
    plt.show()


