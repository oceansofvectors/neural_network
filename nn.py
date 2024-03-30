import numpy as np

from helpers import softmax, categorical_cross_entropy
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def load():
    with open("model.ckpt", "rb") as f:
        return pickle.load(f)


class MultiClassNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = softmax(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def checkpoint(self):
        with open("model.ckpt", "wb") as f:
            f.write(pickle.dumps(self))

    def backward(self, X, y):
        # Assume y is one-hot encoded
        loss_gradient = (self.output - y) / y.shape[0]  # Gradient of categorical cross-entropy
        hidden_gradient = np.dot(loss_gradient, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden)

        d_weights_hidden_output = np.dot(self.hidden.T, loss_gradient)
        d_bias_output = np.sum(loss_gradient, axis=0, keepdims=True)
        d_weights_input_hidden = np.dot(X.T, hidden_gradient)
        d_bias_hidden = np.sum(hidden_gradient, axis=0, keepdims=True)

        # Update parameters
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden
        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output

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

            if epoch % 10 == 0:
                print("Writing checkpoint")
                self.checkpoint()
