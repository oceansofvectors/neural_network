# Simple Feed Forward Neural Network for MNIST Handwritten Digit Recognition

This project implements a simple feed forward neural network using only NumPy as a dependency to predict the MNIST handwritten digit dataset. The trained model achieves an accuracy of 96% on the Kaggle submission with only one hidden layer.

## Creator

Kyle N. Sorensen (sorensen.kyle@pm.me)

## Requirements

- Python 3.x
- NumPy (Note: I kept numpy because they make low level calls to C libraries that optimize matrix math, I don't know how to duplicate that, yet :) )

## Dataset

The model is trained on the MNIST handwritten digit dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).

## Model Architecture

The neural network architecture consists of:
- Input layer: 784 neurons (28x28 flattened image)
- Hidden layer: Configurable number of neurons (default: 128)
- Output layer: 10 neurons (corresponding to the 10 digit classes)

The model uses the sigmoid activation function for the hidden layer and the softmax activation function for the output layer.

## Training

The model is trained using the categorical cross-entropy loss function and gradient descent optimization. The training process is performed for a specified number of epochs and with a configurable batch size.

## Results

The trained model achieves an accuracy of `96%` on the Kaggle competition.
