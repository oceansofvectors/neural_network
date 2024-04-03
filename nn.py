import numpy as np
from helpers import softmax, categorical_cross_entropy
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.input = X
        self.output = self.activation(np.dot(X, self.weights) + self.bias)
        return self.output

    def backward(self, gradient, learning_rate):
        if self.activation == sigmoid:
            activation_gradient = sigmoid_derivative(self.output)
        elif self.activation == softmax:
            activation_gradient = 1
        else:
            raise ValueError("Unsupported activation function")

        gradient = gradient * activation_gradient
        input_gradient = np.dot(gradient, self.weights.T)

        d_weights = np.dot(self.input.T, gradient)
        d_bias = np.sum(gradient, axis=0, keepdims=True)

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return input_gradient


class MultiClassNN:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def checkpoint(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self))

    def backward(self, X, y):
        loss_gradient = (self.layers[-1].output - y) / y.shape[0]  # Gradient of categorical cross-entropy

        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, self.learning_rate)

    def train(self, X, y, epochs=10, batch_size=64):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                output = self.forward(X_batch)
                loss = categorical_cross_entropy(y_batch, output)
                self.backward(X_batch, y_batch)
            if epoch % 1 == 0:
                print(f'Loss at epoch {epoch}: {loss}')

        self.checkpoint(f"model_epoch_{epoch}.ckpt")