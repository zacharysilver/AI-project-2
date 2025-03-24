import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers, learningRate=0.01, epochs=1000):
        self.layers = layers
        self.learningRate = learningRate
        self.epochs = epochs
        self.weights = []
        self.biases = []

        # Random normal initialization for weights (mean 0, small standard deviation)
        for i in range(len(layers) - 1):
            self.weights.append(
                np.random.randn(layers[i], layers[i + 1]) * 0.01
            )  # Small standard deviation
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, x):
        activations = [x]
        zs = []  # Store z values for backpropagation
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward(self, x, y, activations, zs):
        m = x.shape[0]
        deltas = []

        # Output layer error (Binary Cross-Entropy derivative)
        deltas.append(activations[-1] - y)

        # Backpropagate errors
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * sigmoid_derivative(
                activations[i + 1]
            )
            deltas.insert(0, delta)

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= (
                self.learningRate * np.dot(activations[i].T, deltas[i]) / m
            )
            self.biases[i] -= self.learningRate * np.mean(
                deltas[i], axis=0, keepdims=True
            )

    def train(self, x, y):
        for epoch in range(self.epochs):
            activations, zs = self.forward(x)
            self.backward(x, y, activations, zs)

            if epoch % (self.epochs // 10) == 0:
                loss = -np.mean(
                    y * np.log(activations[-1] + 1e-8)
                    + (1 - y) * np.log(1 - activations[-1] + 1e-8)
                )
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, x):
        activations, _ = self.forward(x)
        return (activations[-1] > 0.5).astype(int)


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow


# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)


# Load dataset
data = pd.read_csv("data.csv")
pd.set_option("display.max_columns", None)

# Select relevant features
features = [
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
data = data[features + ["track_popularity"]]

# Convert popularity into a binary classification problem
data["track_popularity"] = (data["track_popularity"] >= 50).astype(int)

x = data.drop(columns=["track_popularity"]).values
y = data[["track_popularity"]].values

# Normalize input features
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Shuffle the data before splitting
shuffle_idx = np.random.permutation(len(x))

# Apply the shuffled indices to both x and y
x_shuffled = x[shuffle_idx]
y_shuffled = y[shuffle_idx]

# Manually split into train/test sets (80% train, 20% test)
splitIdx = int(0.8 * len(x))
xTrain, xTest = x_shuffled[:splitIdx], x_shuffled[splitIdx:]
yTrain, yTest = y_shuffled[:splitIdx], y_shuffled[splitIdx:]


# Define network architectures
architectureList = [
    [x.shape[1], 1],  # 1 perceptron
    [x.shape[1], 5, 1],  # 1 hidden layer
    [x.shape[1], 5, 5, 5, 1],  # 3 hidden layers
]

# Output info and train/evaluate models
for i, arch in enumerate(architectureList):
    print(f"Training Neural Network {i+1} with architecture {arch}:")
    print(f" - Input Layer: {arch[0]} neurons (all track features)")
    for j in range(1, len(arch) - 1):
        print(f" - Hidden Layer {j}: {arch[j]} neurons")
    print(f" - Output Layer: {arch[-1]} neuron (binary classification)\n")

    nn = NeuralNetwork(layers=arch, learningRate=0.1, epochs=50000)
    nn.train(xTrain, yTrain)

    trainPredictions = nn.predict(xTrain)
    testPredictions = nn.predict(xTest)

    trainAccuracy = np.mean(trainPredictions == yTrain)
    testAccuracy = np.mean(testPredictions == yTest)

    print(f"Final Training Accuracy: {trainAccuracy:.4f}")
    print(f"Final Testing Accuracy: {testAccuracy:.4f}\n")
