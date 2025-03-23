import numpy as np
import pandas as pd


# Neural Network class
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, epochs=1000):
        self.layers = layers  # List of layer sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []  # Store weight matrices
        self.biases = []  # Store biases

        # Initialize weights and biases to zero
        for i in range(len(layers) - 1):
            self.weights.append(np.zeros((layers[i], layers[i + 1])))
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            activations.append(a)
        return activations

    def backward(self, X, y, activations):
        m = X.shape[0]  # Number of samples
        errors = [activations[-1] - y]  # Output error

        # Backpropagate errors
        for i in reversed(range(len(self.weights) - 1)):
            errors.insert(
                0,
                np.dot(errors[0], self.weights[i + 1].T)
                * sigmoidDerivative(activations[i + 1]),
            )

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= (
                self.learning_rate * np.dot(activations[i].T, errors[i]) / m
            )
            self.biases[i] -= self.learning_rate * np.mean(
                errors[i], axis=0, keepdims=True
            )

    def train(self, X, y):
        for epoch in range(self.epochs):
            activations = self.forward(X)
            self.backward(X, y, activations)
            if epoch % (self.epochs // 10) == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        return (self.forward(X)[-1] > 0.5).astype(int)


# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sigmoid activation function derivative
def sigmoidDerivative(x):
    return x * (1 - x)


# load dataset
data = pd.read_csv("data.csv")
pd.set_option("display.max_columns", None)

data = data[
    [
        "track_popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]
]
data["track_popularity"] = (data["track_popularity"] >= 50).astype(int)

X = data.drop(columns=["track_popularity"]).values
y = data[["track_popularity"]].values

# Normalize input features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test networks
architectures = [
    [X.shape[1], 1],  # 1 perceptron
    [X.shape[1], 5, 1],  # 1 hidden layer
    [X.shape[1], 5, 5, 5, 1],  # 3 hidden layers
]

# output info
for i, arch in enumerate(architectures):
    print(f"Training Neural Network {i+1} with architecture {arch}:")
    print(f" - Input Layer: {arch[0]} neurons (all track features)")
    for j in range(1, len(arch) - 1):
        print(f" - Hidden Layer {j}: {arch[j]} neurons")
    print(f" - Output Layer: {arch[-1]} neuron (binary classification)\n")

    nn = NeuralNetwork(layers=arch, learning_rate=0.01, epochs=1000)

    nn.train(X, y)

    predictions = nn.predict(X)

    accuracy = np.mean(predictions == y)
    print(f"Final Accuracy: {accuracy:.4f}\n")
