import numpy as np


class Perceptron:
    def __init__(self, n_iter=100, random_state=1, lr=0.01) -> None:
        self.n_iter = n_iter
        self.random_state = random_state
        self.lr = lr

    def fit(self, X, y):
        X = self._include_bias(X)
        random_gen = np.random.RandomState(seed=self.random_state)
        self.weights = random_gen.normal(loc=0.0, scale=1.0, size=X.shape[1])

        for _ in range(self.n_iter):
            for sample, target in zip(X, y):
                prediction = self.predict(sample)
                update = self.lr * (target - prediction)
                self.weights += update * sample
    
    def _include_bias(self, X):
        return np.append(np.ones((len(X), 1)), X, axis=1)
                
    def input_net(self, X):
        return np.dot(X, self.weights) #+ self.weights[0]

    def predict(self, X):
        if X.shape[0] != self.weights.shape[0]:
            X = self._include_bias(X)
        return np.where(self.input_net(X) >= 0, 1, -1)

    
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split


    model = Perceptron()
    X, y = make_classification(n_samples=1500, n_features=2, n_classes=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    correct = 0
    for pred, target in zip(predictions, y_test):
        if pred == target:
            correct += 1
    print(f"Accuracy: {correct/y_test.shape[0]:.2f}")
    

