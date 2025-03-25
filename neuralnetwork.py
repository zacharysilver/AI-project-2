import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers, learningRate=0.01, epochs=1000, patience=10):
        self.layers = layers
        self.learningRate = learningRate
        self.epochs = epochs
        self.patience = patience  # Early stopping patience
        self.weights = []
        self.biases = []

        # Random normal initialization for weights
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward(self, x):
        activations = [x]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward(self, x, y, activations, zs):
        m = x.shape[0]
        deltas = [activations[-1] - y]

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * sigmoidDerivative(activations[i + 1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learningRate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learningRate * np.mean(deltas[i], axis=0, keepdims=True)

    def train(self, x, y):
        bestLoss = float("inf")
        patienceCounter = 0

        for epoch in range(self.epochs):
            activations, zs = self.forward(x)
            loss = -np.mean(y * np.log(activations[-1] + 1e-8) + (1 - y) * np.log(1 - activations[-1] + 1e-8))
            self.backward(x, y, activations, zs)

            # Print progress
            if epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

            # Early stopping check
            if np.abs(loss - bestLoss) < 1e-3:  # Relaxed improvement threshold
                patienceCounter += 1
            else:
                patienceCounter = 0  # Reset patience counter if loss improves
                bestLoss = loss

            if patienceCounter >= self.patience:
                print(f"Stopping early at epoch {epoch} (Loss: {loss:.4f})\n")
                break


    def predict(self, x):
        activations, _ = self.forward(x)
        return (activations[-1] > 0.5).astype(int)


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# Derivative of sigmoid function
def sigmoidDerivative(x):
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

# Shuffle data
shuffleIdx = np.random.permutation(len(x))
x, y = x[shuffleIdx], y[shuffleIdx]

# Define network architectures
architectureList = [
    [x.shape[1], 1],  # Single perceptron
    [x.shape[1], 5, 1],  # One hidden layer
    [x.shape[1], 5, 5, 5, 1],  # Three hidden layers
]

# 5-Fold Cross Validation
numFolds = 5
foldSize = len(x) // numFolds

for i, arch in enumerate(architectureList):
    print(f"\nTraining Neural Network {i+1} with architecture {arch} using 5-fold cross-validation:")

    accuracies = []

    for fold in range(numFolds):
        print("\n")
        print(f"Fold {fold + 1}/{numFolds}")

        # Split into training and validation sets
        start, end = fold * foldSize, (fold + 1) * foldSize
        xVal, yVal = x[start:end], y[start:end]
        xTrain = np.vstack((x[:start], x[end:]))
        yTrain = np.vstack((y[:start], y[end:]))

        # Train model
        nn = NeuralNetwork(layers=arch, learningRate=0.1, epochs=50000, patience=1000)
        nn.train(xTrain, yTrain)

        # Evaluate model
        predictions = nn.predict(xVal)
        tp = sum(1 if (a == 1) and (b == 1) else 0 for a, b in zip(yVal, predictions))
        fp = sum(1 if (a == 0) and (b == 1) else 0 for a, b in zip(yVal, predictions))
        tn = sum(1 if (a == 0) and (b == 0) else 0 for a, b in zip(yVal, predictions))
        fn = sum(1 if (a == 1) and (b == 0) else 0 for a, b in zip(yVal, predictions))


        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        accuracy = np.mean(predictions == yVal)
        accuracies.append(accuracy)

        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    # Compute mean and standard deviation of accuracy across folds
    meanAccuracy = np.mean(accuracies)
    stdAccuracy = np.std(accuracies)

    print(f"\nFinal Results for Neural Network {i+1}:")
    print(f"Mean Accuracy: {meanAccuracy:.4f}")
    print(f"Standard Deviation of Accuracy: {stdAccuracy:.4f}\n")
